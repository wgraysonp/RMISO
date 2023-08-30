from sklearn.datasets import fetch_covtype
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import numpy as np
from graph_structure.data_graph import DataGraph


class CovType(Dataset):

    def __init__(self, train=True, zero_one=True):
        self.train = train
        # label samples as either 0 or 1 if true and -1 or 1 if false
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


def test():
    data = CovType(train=True)
    G = DataGraph(data, num_nodes=100, num_edges=11, topo='cycle', sep_classes=True)
    print(G.nodes[0]['data'])
    l = 0
    for node in G.nodes:
        l += len(G.nodes[node]['data'])
    print(l)
    #for idx in G.nodes[50]['data']:
        #print(data.targets[idx])


if __name__ == "__main__":
    test()

