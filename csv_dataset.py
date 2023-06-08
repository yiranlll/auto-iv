import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import glob
import random

class CustomCSVDataset(Dataset):
    def __init__(self, filenames, batch_size):
        self.filenames = filenames 
        self.batch_size = batch_size # `batch_size` number of files make a batch

    def __len__(self):
        return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, index):
        batch = self.filenames[index * self.batch_size:(index+1) * self.batch_size]
        v, x, y = [], [], []
        for file in batch:
            df = pd.read_csv(open(file, 'r'), skiprows=1)
            v.append(df.values[:, 5:11].astype(np.float32))
            x.append(df.values[:, 12:14].astype(np.float32))
            y.append(df.values[:, 4].astype(np.float32))
        v = np.concatenate(v)
        x = np.concatenate(x)
        y = np.concatenate(y)
        y = y.reshape(len(y), 1)

        if index == self.__len__():  
            raise IndexError

        return v, x, y

if __name__ == "__main__":
    files = glob.glob("data/*.csv")
    print(files[:5])
    dataset = CustomCSVDataset(files, 4)
    print("Length of Dataset: {}".format(len(dataset)))
    print("\nData preview:")
    record = next(iter(dataset))
    print("\nv:\t")
    print(record['v'])
    print("\nx:\t")
    print(record['x'])
    print("\ny:\t")
    print(record['y'])