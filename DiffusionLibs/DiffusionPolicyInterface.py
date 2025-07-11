import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from IPython.display import clear_output

from .DiffusionQ import DiffusionQLearner

class DiffusionPolicyInterface:
    def __init__(self, 
                 envParams,
                 max_episode_steps: int = 5000,
                 device: str = "cuda",
                 gamma: float = 0.99,
                 tau: float = 0.1,
                 lr: float = 5e-2,
                 Lq_weight: float = 1.0,
                 Ld_weight: float = 1.0,
                 N_action_candidates: int = 20,
                 iql_tau: float = 0.1,
                 temperature: float = 10.0,
                 ):
        self.n_users = envParams['N_user']
        self.max_episode_steps = max_episode_steps
        self.alpha_range = envParams['alpha_range']
        self.bandwidth = envParams['B']
        self.len_window = envParams['LEN_window']
        self.device = device
        self.state_dim = self.n_users # [u_i | i=0,1,...,N-1]
        self.action_dim = 2 * self.n_users + 2 # [w_i, r_i, M, alpha | i=0,1,...,N-1]
        self.diffusionQ = DiffusionQLearner(
            self.state_dim, self.action_dim, 
            gamma=gamma, tau=tau, lr=lr, Lq_weight=Lq_weight, Ld_weight=Ld_weight, 
            N_action_candidates=N_action_candidates,
            iql_tau=iql_tau, temperature=temperature, device=device)  
        self.N_action_candidates = N_action_candidates    
        
    def _from_diffusionQ_action_to_env_action(self, action):
        """Convert RL action to policy parameters (w, r, M, alpha)."""
        """The action is normalized to [-1, 1]"""
        # Convert normalized [-1, 1] actions to actual values
        action = np.clip(action, -1, 1)
        (w_norm, r_norm, M_norm, alpha_norm) = self._split_action(action)
        # Convert w from [-1, 1] to binary (threshold at 0)
        w = (w_norm > 0.0).astype(int)
        
        # Convert r from [-1, 1] to [0, bandwidth]
        r_raw = (r_norm + 1.0) / 2.0 * self.bandwidth  # [0, bandwidth]
        r = np.clip(r_raw, 0, self.bandwidth)
        
        # Convert M from [-1, 1] to [1, min(10, len_window)]
        M_raw = (M_norm + 1.0) / 2.0 * (min(10, self.len_window) - 1) + 1
        M = int(np.clip(M_raw, 1, min(10, self.len_window)))
        
        # Convert alpha from [-1, 1] to [alpha_min, alpha_max]
        alpha_norm = (alpha_norm + 1.0) / 2.0  # Convert to [0, 1]
        alpha = self.alpha_range[0] + alpha_norm * (self.alpha_range[1] - self.alpha_range[0])
        
        # Normalize r to respect total bandwidth constraint
        total_r = np.sum(r)
        if total_r > self.bandwidth:
            r = r * (self.bandwidth / total_r)  # Scale down proportionally
        
        return w, r, M, alpha
    
    def _from_env_action_to_diffusionQ_action(self, action):
        # --- 1) invert w: {0,1} → [-1,1]  --------------------------
        #    we map 0 → -1, 1 → +1
        action = np.concatenate([np.atleast_1d(a) for a in action])
        (w, r, M, alpha) = self._split_action(action)
        w_norm = 2*np.array(w).astype(float) - 1.0

        # --- 2) invert r: [0, bandwidth] → [-1,1] ----------------
        # original: r = (r_norm+1)/2 * bandwidth
        r_norm = 2.0 * (r / self.bandwidth) - 1.0
        # clip back into [-1,1] in case of numerical drift
        r_norm = np.clip(r_norm, -1.0, 1.0)

        # --- 3) invert M: [1, M_max] → [-1,1] ---------------------
        M_max = min(10, self.len_window)
        # original: M = (M_norm+1)/2*(M_max-1) + 1
        M_norm = 2.0 * ((float(M) - 1.0) / (M_max - 1.0)) - 1.0
        M_norm = np.clip(M_norm, -1.0, 1.0)

        # --- 4) invert alpha: [α_min, α_max] → [-1,1] -------------
        alpha_min, alpha_max = self.alpha_range
        # original: alpha = α_min + ((alpha_norm+1)/2) * (α_max - α_min)
        alpha_unit = (alpha - alpha_min) / (alpha_max - alpha_min)   # in [0,1]
        alpha_norm = 2.0 * alpha_unit - 1.0
        alpha_norm = np.clip(alpha_norm, -1.0, 1.0)

        # --- 5) pack everything back into one vector -------------
        action_norm = np.concatenate([
            w_norm,             # length = n_users
            r_norm,             # length = n_users
            np.array([M_norm, alpha_norm])
        ])
        return action_norm

    def _split_action(self, action):
        w = action[:self.n_users]
        r = action[self.n_users:2*self.n_users]
        M = action[2*self.n_users]
        alpha = action[2*self.n_users + 1]
        return w, r, M, alpha

    def _preprocess_action(self, action):
        A = []
        for a in action:
            a_DQ = self._from_env_action_to_diffusionQ_action(a)
            A.append(a_DQ)
        return np.array(A)
    
    def _preprocess_state(self, state):
        # [0, LEN_window] → [-1, 1]
        S = 2.0*np.array(state) / self.len_window - 1.0
        return S
    
    #==============================================
    # ==== IMPORTANT FUNCTIONS ====================
    #==============================================
    # The modified reward now becomes 1-packet loss, so Q 
    # learning try to "maximize" the reward 
    def _preprocess_reward(self, reward, action):
        w, r, _, alpha = zip(*action)
        w, r, alpha = np.array(w), np.array(r), np.array(alpha)
        cost = np.sum(w * r, axis=1)
        excess  = cost - alpha * self.bandwidth
        penalty = (np.maximum(excess, 0.0)/self.bandwidth)**2
        #--------------------------
        R = 1.0-np.array(reward) - 0.0*penalty
        return R
    
    def _observation_mode(self, u, u_predicted, obvMode):
        if obvMode == "perfect":
            return u
        elif obvMode == "predicted":
            return u_predicted
        else:
            raise ValueError(f"Invalid observation mode: {obvMode}")

    def sample(self, state: np.ndarray, N_action_candidates: int = 10, sample_method: str = "mean"):
        s = self._preprocess_state(state)
        s = torch.as_tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
        a = self.diffusionQ.sample(s, N_action_candidates, sample_method).cpu().detach().numpy()
        return self._from_diffusionQ_action_to_env_action(a[-1])
    
    def train(self, data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              env=None,
              batch_size: int = 1024,
              epochs: int = 10,
              iql_flag: bool = False,
              sample_method: str = "mean",
              verbose: bool = True):

        (state, action, reward, state_next) = data
        S = self._preprocess_state(state)
        A = self._preprocess_action(action)
        R = self._preprocess_reward(reward, action)
        S_next = self._preprocess_state(state_next)
        dataset = TensorDataset(
            torch.as_tensor(S).to(self.device), torch.as_tensor(A).to(self.device),
            torch.as_tensor(R).to(self.device), torch.as_tensor(S_next).to(self.device), 
        )
        
        LdRecord = []
        LqRecord = []
        lossCriticRecord = []
        estRewardRecord = []
        best_Lq = np.inf
        for ep in range(1, epochs + 1):
            self.diffusionQ.train()
            loader = DataLoader(dataset, batch_size=min(batch_size, len(S)), shuffle=True, drop_last=True)
            with tqdm(loader, desc=f'Epoch {ep}/{epochs}', unit='batch', leave=False) as batch_bar:
                for batch in batch_bar:
                    Ld, Lq, loss_critic, est_reward = self.diffusionQ.update(batch, iql_flag=iql_flag)
                    batch_bar.set_postfix({'Ld': f'{Ld:.6f}', 'Lq': f'{Lq:.6f}', 'critic': f'{loss_critic:.6f}', 'est_reward': f'{est_reward:.6f}'})
            LdRecord.append(Ld)
            LqRecord.append(Lq)
            lossCriticRecord.append(loss_critic)
            estRewardRecord.append(est_reward)
            if Lq < best_Lq:
                best_Lq = Lq
                model_state_dict = self.diffusionQ.state_dict()
            if ep % (epochs/10) == 0 and verbose == True:
                if env is not None:
                    self.diffusionQ.eval()
                    evalResult = self.eval(env, 
                                           num_windows=500, 
                                           obvMode="predicted", 
                                           mode="test", 
                                           type="data", 
                                           sample_method=sample_method)
                    avgReward = np.mean(evalResult['rewardRecord'])
                else:
                    avgReward = np.nan
                tqdm.write(f"Epoch {ep:4d}/{epochs:4d}  Avg Ld={np.mean(LdRecord[-int(epochs/10):]):.6f}" + 
                           f"  Avg Lq={np.mean(LqRecord[-int(epochs/10):]):.6f}" + 
                           f"  Avg loss_critic={np.mean(lossCriticRecord[-int(epochs/10):]):.6f}" +
                           f"  Test packet loss={avgReward:.4f}" +
                           f"  Avg est_reward={np.mean(estRewardRecord[-int(epochs/10):]):.6f}")
        info = {
            'LdRecord': LdRecord,
            'LqRecord': LqRecord,
            'lossCriticRecord': lossCriticRecord,
            'estRewardRecord': estRewardRecord,
            'R': R,
            'S': S,
            'A': A,
            'S_next': S_next
        }
        return model_state_dict, info
    
    def _step(self, env, obvMode, sample_method="mean"):
        u, u_predicted = env.getStates()
        u_active = self._observation_mode(u, u_predicted, obvMode)
        (w, r, M, alpha) = self.sample(u_active, 
                                       N_action_candidates=self.N_action_candidates, 
                                       sample_method=sample_method)
        reward = env.applyActions(np.array(w), np.array(r), M, alpha)
        env.updateStates()
        u_next, u_next_predicted = env.getStates()
        u_next_active = self._observation_mode(u_next, u_next_predicted, obvMode)
        return u_active, (w, r, M, alpha), reward, u_next_active

    def eval(self, env, num_windows=1000, obvMode="perfect", mode="test", type="data", sample_method="mean"):
        env.reset()
        env.selectMode(mode=mode, type=type)
        rewardRecord = []   
        actionsRecord = []
        uRecord = []
        uNextRecord = []
        for window in tqdm(range(num_windows), desc="Evaluation windows"):
            (u_active, action, reward, u_next_active) = self._step(env, obvMode, sample_method)
            (w, r, M, alpha) = action
            #============ Record Results ============
            rewardRecord.append(reward)
            actionsRecord.append((np.array(w), np.array(r), M, alpha))
            uRecord.append(u_active)
            uNextRecord.append(u_next_active)

        evalResult = {
            "rewardRecord": rewardRecord,
            "actionsRecord": actionsRecord,
            "uRecord": uRecord,
            "uNextRecord": uNextRecord
        }
        return evalResult

    def train_online(self, env,
              batch_size: int = 128,
              epochs: int = 10,
              iql_flag: bool = False,
              sample_method: str = "mean",
              obvMode: str = "perfect",
              num_windows: int = 1000,
              mode: str = "test",
              type: str = "data",
              verbose: bool = True):
        env.reset()
        env.selectMode(mode=mode, type=type)

        LdRecord = []
        LqRecord = []
        lossCriticRecord = []
        with tqdm(range(num_windows), desc="Training windows") as window_bar:
            uData, aData, rData, uNextData = [], [], [], []
            for window in window_bar:
                (u_active, action, reward, u_next_active) = self._step(env, obvMode, sample_method)
                uData.append(u_active); aData.append(action); rData.append(reward); uNextData.append(u_next_active)
                dataset = (uData, aData, rData, uNextData)
                if window % batch_size == 0:
                    model_state_dict, info =self.train(data=dataset,
                        env=env,
                        batch_size=batch_size,
                        epochs=epochs,
                        iql_flag=iql_flag,
                        sample_method=sample_method,
                        verbose=verbose)
                    LdRecord += info['LdRecord']
                    LqRecord += info['LqRecord']
                    lossCriticRecord += info['lossCriticRecord']
                window_bar.set_postfix({'Ld': f'{np.mean(LdRecord[-int(num_windows/batch_size):]):.6f}',
                                        'Lq': f'{np.mean(LqRecord[-int(num_windows/batch_size):]):.6f}', 
                                        'loss_critic': f'{np.mean(lossCriticRecord[-int(num_windows/batch_size):]):.6f}'})
                uData, aData, rData, uNextData = [], [], [], []
        return model_state_dict, info
        