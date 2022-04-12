import pandas as pd
import numpy as np

class DS():
    
    def __init__(self) -> None:
        
        # Load X
        data_csv = pd.read_csv('./data/train.csv')
        data = data_csv.values
        data = data[:, 3:]

        # normalize
        data = data - data.mean(axis=0)
        data = data / data.std(axis=0)

        X = data.reshape((int(data.shape[0]/60), 60, 13))

        # Load t
        target_csv = pd.read_csv('./data/train_labels.csv')
        target = target_csv.values[:,1:]
        t = target

        # Make t one-hot
        t = np.array([
            (t == 0).astype(int),
            (t == 1).astype(int)
        ]).T.squeeze().astype(int)

        self.X = X
        self.t = t

    def get_data(self):
        return (self.X, self.t)