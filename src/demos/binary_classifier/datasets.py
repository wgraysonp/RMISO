from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
import torch
from torch.utils.data import Dataset
import numpy as np
from graph_structure.data_graph import DataGraph
import networkx as nx
import matplotlib.pyplot as plt


class CovType(Dataset):

    def __init__(self, train=True, zero_one=True):
        self.train = train
        # label samples as either 0 or 1 if true and -1 or 1 if false
        self.p = 54
        self.zero_one = zero_one
        self._load_and_preprocess_data()
        self.length = self.features.shape[0]
        self.classes = [0, 1] if zero_one else [-1, 1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def _load_and_preprocess_data(self):
        X, y = fetch_covtype(data_home="data", random_state=0, shuffle=False, return_X_y=True)
        if self.zero_one:
            y = np.array(list(map(lambda x: 1 if x == 2 else 0, y)))
        else:
            y = np.array(list(map(lambda x: 1 if x == 2 else -1, y)))
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=450000)
        if self.train:
            self.features = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            self.features = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)

class W8a(Dataset):
    def __init__(self, train=True, zero_one=False):
        self.train = train
        self.zero_one = zero_one
        self.p = 300
        self._load_and_preprocess_data()
        self.length = self.features.shape[0]
        self.classes = [0, 1] if zero_one else [-1, 1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def _load_and_preprocess_data(self):
        X, y = fetch_openml(name='w8a', data_home='data', return_X_y=True)
        X = X.todense()
        if self.zero_one:
            y = np.array(list(map(lambda x: 0 if x == -1 else 1, y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=50000)
        if self.train:
            self.features = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            self.features = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


class Synthetic(Dataset):
    torch.manual_seed(78)

    def __init__(self, train=True, zero_one=False, p=200):
        self.train = train
        self.zero_one = zero_one
        self.p = p
        self._load_and_preprocess_data()
        self.length = self.features.shape[0]
        self.classes = [0, 1] if zero_one else [-1, 1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def _load_and_preprocess_data(self):
        n_data = 6000
        n_train = 5000
        X = torch.randn((n_data, self.p))
        x0 = torch.rand(self.p)
        z = torch.rand(n_data)
        y = torch.zeros(n_data)
        for i in range(n_data):
            b = torch.dot(x0, X[i])
            a = 1/(1 + torch.exp(-b))
            y[i] = 1 if z[i] <= a else -1

        if self.train:
            self.features = X[:n_train, :]
            self.targets = y[:n_train].reshape(-1, 1)
        else:
            self.features = X[n_train:, :]
            self.targets = y[n_train:].reshape(-1, 1)


class A9a(Dataset):
    def __init__(self, train=True, zero_one=False):
        self.train = train
        self.zero_one = zero_one
        self.p = 123
        self._load_and_preprocess_data()
        self.length = self.features.shape[0]
        self.classes = [0, 1] if zero_one else [-1, 1]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

    def _load_and_preprocess_data(self):
        X, y = fetch_openml(name='a9a', data_home='data', return_X_y=True)
        X = X.todense()
        if self.zero_one:
            y = np.array(list(map(lambda x: 0 if x == -1 else 1, y)))
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2, train_size=35000)
        if self.train:
            self.features = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        else:
            self.features = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1)


def test():
    data = A9a()
    avg_n = 0
    for i in range(len(data.features)):
        n = np.linalg.norm(data.features[i])
        avg_n += n
    avg_n *= 1/(len(data.features))

    print(avg_n)


if __name__ == "__main__":
    test()

