from torch.utils.data import Dataset
from data import DS

class Tab22_Dataset(Dataset):

    def __init__(self) -> None:
        
        data = DS()
        self.X = data.X
        self.t = data.t

    def __len__(self) -> int:
        
        return self.X.shape[0]

    def __getitem__(self, index):
        
        return (self.X[index], self.t[index])