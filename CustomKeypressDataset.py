import os
import pandas as pd
from torch.utils.data import Dataset

# Keypress dataset
class CustomKeypressDataset(Dataset):
    # Labels (press/no press)
    # Directory of sensor data
    def __init__(self, annotations_file, kp_dir):
        self.kp_labels = pd.read_csv(annotations_file)
        self.kp_dir = kp_dir

    # Dataset size
    def __len__(self):
        return len(self.kp_labels)

    # Read next pair of sensor data/label
    def __getitem__(self, idx):
        kp_path = os.path.join(self.kp_dir, self.kp_labels.iloc[idx, 0])
        f = open(kp_path, 'r')
        kpraw = f.read()
        keypress = map(int, kpraw.split(","))
        label = self.kp_labels.iloc[idx, 1]
        return keypress, label
