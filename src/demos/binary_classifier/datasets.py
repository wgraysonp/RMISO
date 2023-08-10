from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np


class CovType(Dataset):

    def __init__(self, train=True):
        self.train = train
        self._load_and_preprocess_data()
        self.length = self.features.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def _load_and_preprocess_data(self):
        X, y = fetch_covtype(data_home="data", random_state=0, shuffle=True, return_X_y=True)
        y = np.array(list(map(lambda x: 1 if x == 2 else 0, y)))
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
        if self.train:
            self.features = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            self.features = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


def test():
    data = CovType(train=True)
    print(len(data))
    print(data.features[1].shape[0])


if __name__ == "__main__":
    test()

