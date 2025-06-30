import pandas as pd
import matplotlib.pyplot as plt

class DatasetReader:
    def __init__(self, parentFolder):
        self.dfRaw = []
        self.dfLibrary = {}
        self.readFileIndexsList = []
        self.parentFolder = parentFolder
        self.configuration()

    def configuration(self):
        self.readFileIndexsList = ["0", "1", "2", "3", "4","5","6","7","8","9","10"]

    def readRawDataset(self, filename, datasetLabel):
        df_read = pd.read_csv(filename)
        dfRaw = processTimeseriesData(df_read)
        dfRaw = removeOutlier(dfRaw)
        if datasetLabel not in self.dfLibrary:
            self.dfLibrary[datasetLabel] = dfRaw
        else:
            self.dfLibrary[datasetLabel] = concatenateDataframes(self.dfLibrary[datasetLabel], dfRaw)

    def visualization(self, taskName, fingerName, idxsObv=1000, figsize=(10, 5)):
        if fingerName in ["Thumb", "Index", "Middle", "Center"] is False:
            ValueError("Invlid Finger name.")
        # Create the figure and subplots
        fig = plt.figure(figsize=figsize)  # Set the figure size
        ax0 = fig.add_subplot(121)  # First subplot (1 row, 2 columns, 1st position)
        ax1 = fig.add_subplot(122, projection='3d')  # Second subplot (1 row, 2 columns, 2nd position)

        # Plot the first subplot
        ax0.plot(self.dfLibrary[taskName][f"{fingerName}Force"][0:idxsObv])
        ax0.set_xlabel("Index")
        ax0.set_ylabel("Force")
        ax0.set_title("Force (Thumb)")

        # Plot the second subplot
        x = self.dfLibrary[taskName][f"{fingerName}X"][0:idxsObv]
        y = self.dfLibrary[taskName][f"{fingerName}Y"][0:idxsObv]
        z = self.dfLibrary[taskName][f"{fingerName}Z"][0:idxsObv]
        ax1.plot(x, z, y)
        ax1.set_xlabel('X Axis')
        ax1.set_ylabel('Y Axis')
        ax1.set_zlabel('Z Axis')
        ax1.set_title("Position (Thumb)")

        # Adjust layout
        plt.tight_layout()  # Ensures no overlap between subplots
        plt.show()

def removeOutlier(df):
    # 1. Calculate Q1, Q3, and IQR for each column
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    # 2. Define the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # 3. Replace outliers with NaN
    df_mask = df.mask((df < lower_bound) | (df > upper_bound))
    df_cleaned = df_mask.interpolate(method='linear')
    df_out = df_cleaned
    return df_out.copy()

def processTimeseriesData(
        df: pd.DataFrame, 
        head_discarding_ratio=0.05, tail_discarding_ratio=0.95) -> pd.DataFrame:
    try:
        # Skip the first row if it contains headers
        if df.iloc[0, 0] == "Time":
            df = df.iloc[1:].copy()

        # Apply conversion
        df["Time"] = df["Time"].apply(
            lambda x: int(x.split(':')[0]) * 3600 +  # Hours to seconds
            int(x.split(':')[1]) * 60 +   # Minutes to seconds
            float(x.split(':')[2])        # Seconds and milliseconds
        )
        head_idxs = int(df.shape[0]*head_discarding_ratio)
        tail_idxs = int(df.shape[0]*tail_discarding_ratio)
        df = df[head_idxs:tail_idxs]

        # Set initial time to 0 and calculate relative times
        initial_time = df["Time"].iloc[0].copy()
        #df["Time"] = df["Time"] - initial_time
        df.loc[:, "Time"] = df["Time"] - initial_time

        return df.reset_index(drop=True)
    except Exception as e:
        print(f"An error occurred during the transformation: {e}")
    return df

def concatenateDataframes(df1, df2):
    """
    Concatenates two dataframes with a time column in milliseconds as the first column.
    Adjusts the time in the second dataframe to ensure consistent time progression in the concatenated result.

    Parameters:
    - df1: The first pandas DataFrame.
    - df2: The second pandas DataFrame.

    Returns:
    - A concatenated DataFrame with consistent time in the first column.
    """
    # Ensure the 'time' column is in numeric format (float for ms)
    df1.iloc[:, 0] = pd.to_numeric(df1.iloc[:, 0], errors='coerce')
    df2.iloc[:, 0] = pd.to_numeric(df2.iloc[:, 0], errors='coerce')

    # Calculate the time offset to adjust df2's time column
    time_offset = df1.iloc[-1, 0] - df2.iloc[0, 0] + 0.1  # Adding 0.1 ms to avoid overlap
    
    # Apply the offset to df2's time column
    df2_adjusted = df2.copy()
    df2_adjusted.iloc[:, 0] += time_offset

    # Concatenate df1 and df2_adjusted
    concatenated_df = pd.concat([df1, df2_adjusted], ignore_index=True)
    
    return concatenated_df
