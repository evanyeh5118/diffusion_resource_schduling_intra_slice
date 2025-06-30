import torch
import torch.nn as nn
import torch.optim as optim

class DeadFeaturesToTrafficLayer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_classes, target_len_seq, num_layers=1, dropout_rate=0.5):
        super(DeadFeaturesToTrafficLayer, self).__init__()
        self.num_classes = num_classes
        self.target_len_seq = target_len_seq
        # Input layer with batch normalization and dropout
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),  # Batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        # Hidden layers with residual connections, batch normalization, and dropout
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),  # Batch normalization
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ))
        # Output layer
        self.trans2transmission_layer = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()
        self.trans2traffic_layer = nn.Linear(hidden_size + self.target_len_seq, 1)
        self.trans2trafficClass_layer = nn.Linear(hidden_size + self.target_len_seq, self.num_classes)
        # Apply weight initialization
        self._initialize_weights()

    def forward(self, x):
        # Initial layer
        x = self.input_layer(x)
        residual = x  # Save initial input as residual for skip connections
        
        # Process through hidden layers with skip connections
        for layer in self.hidden_layers:
            out = layer(residual)
            residual = out + residual  # Add skip connection if dimensions match

        # Output layer
        x_transmission = self.sigmoid(self.trans2transmission_layer(residual))
        cat_features = torch.cat((x_transmission, residual), -1)
        x_traffic = self.trans2traffic_layer(cat_features)
        x_traffic_class = self.trans2trafficClass_layer(cat_features)
        
        return x_traffic, x_traffic_class, x_transmission

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

class TrafficPredictorContextAssisted(nn.Module):
    def __init__(self, 
                 input_size, hidden_size, output_size, num_classes,
                 len_source, len_target, dt, degree, device, num_layers=1, dropout_rate=0.5):
        super(TrafficPredictorContextAssisted, self).__init__()  
        self.M = self.ComputePolyMatrix(len_source, len_target, dt, degree, device)
        self.len_dbf = self.compute_feature_length(len_target+1)
        self.dbf2traffic = DeadFeaturesToTrafficLayer(
            self.len_dbf, hidden_size, output_size, num_classes, len_target, 
            num_layers=num_layers, dropout_rate=dropout_rate
        ).to(device)
        self.reluOut = nn.ReLU()
        
    def Create_matrix(self, dt: float, L: int, degree: int) -> torch.Tensor:
        n_values = torch.arange(L, dtype=torch.float32)
        matrix = torch.stack([(n_values * dt) ** d for d in range(degree + 1)], dim=1)
        return matrix

    def ComputePolyMatrix(self, len_source, len_target, dt, degree, device):
        # Create matrices for source and prediction
        A_s = self.Create_matrix(dt, len_source, degree)  # Shape: (len_source, degree + 1)
        A_p = self.Create_matrix(dt, len_source + len_target, degree)[len_source:]  # Shape: (len_target, degree + 1)
        A_s = A_s.to(device)
        A_p = A_p.to(device)

        A_s_t = A_s.T
        M = A_p @ torch.linalg.inv(A_s_t @ A_s) @ A_s_t
        return M

    def ComputeDeadbandFeatures(self, data):
        # data: (T, B, F)
        T, B, F = data.shape
        mag = torch.sqrt((data ** 2).sum(dim=2))
        data_b = data.permute(1, 0, 2)
        mag_sq_b = (data_b ** 2).sum(dim=2, keepdim=True)
        dist_sq_b = mag_sq_b + mag_sq_b.transpose(1, 2) - 2 * (data_b @ data_b.transpose(1, 2))
        dist_b = torch.sqrt(dist_sq_b.clamp_min(1e-12))  # (B,T,T)
        i_idx, j_idx = torch.tril_indices(T, T, offset=-1)
        pairwise_features_b = dist_b[:, i_idx, j_idx]
        mag_b = mag.transpose(0, 1)  # (B,T)
        features_b = torch.cat([mag_b, pairwise_features_b], dim=1)

        return features_b

    def compute_feature_length(self, data_length):
        if data_length < 2:
            raise ValueError("Data length must be at least 2 to compute pairwise distances.")
        # Magnitudes + Pairwise distances
        feature_length = data_length + (data_length * (data_length - 1)) // 2
        return feature_length

    def forward(self, src, last_trans_src):
        # src: [src_len, batch_size, input_dim]
        # size: (len_seq, batch_size, dim)
        motion_predict = (self.M.unsqueeze(0) @ src.permute(2, 0, 1)).permute(1, 2, 0)
        motion_feature = torch.cat([motion_predict, last_trans_src], dim=0)
        #motion_feature = torch.cat([src, last_trans_src], dim=0)
        db_features = self.ComputeDeadbandFeatures(motion_feature)
        traffic_est, traffic_class_est, transmission_est = self.dbf2traffic(db_features)
        traffic_est = self.reluOut(traffic_est)
        return traffic_est, traffic_class_est, transmission_est, motion_predict 
    
class CustomLossFunction(nn.Module):
    def __init__(self, lambda_trans=0.1, lambda_class=0.1):
        super(CustomLossFunction, self).__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.lambda_trans = lambda_trans
        self.lambda_class = lambda_class

    def forward(self, 
        outputs_traffic, traffics,
        outputs_traffic_class, traffic_class,
        outputs_transmissions, transmissions):
        mse_loss = self.mse(outputs_traffic, traffics)
        ce_loss = self.cross_entropy(outputs_traffic_class, traffic_class)
        bce_loss = self.bce(outputs_transmissions, transmissions)       

        return mse_loss + self.lambda_class*ce_loss + self.lambda_trans*bce_loss, mse_loss