from torch.utils.data import Dataset
from torch.utils.data import random_split

class DataPrep(Dataset):
    def __init__(self, df):
        self.X = df[:, :-1]
        self.y = df[:, -1]
        #self.y = self.y.reshape((len(self.y), 1))

    def split(self, test_ratio):
        test_size = round(test_ratio * len(self.X))
        train_size = len(self.X) - test_size
        return random_split(self, [train_size, test_size])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]
