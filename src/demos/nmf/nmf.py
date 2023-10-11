'''
adapted from https://github.com/HanbaekLyu/ONMF_ONTF_NDL/blob/master/src/onmf.py
'''

import numpy as np
from numpy import linalg as LA


class NMFOptim:
    def __init__(self, W, X=None, H=None, n_nodes=100, n_components=100, alpha=0):
        self.W = W
        self.X = X
        self.H = H
        self.n_nodes = n_nodes
        self.n_components = n_components
        self.alpha = alpha
        self.curr_node = 0

    def set_data_matrix(self, X):
        self.X = X

    def set_code_matrix(self, H):
        self.H = H

    def get_code_matrix(self):
        return self.H

    def set_curr_node(self, node_id):
        self.curr_node = node_id

    def update_code(self, H0, sub_iter=1, stopping_diff=0.1, eta=1.0):
        '''
        find H = argmin_{\R \geq 0} (|X - WH| + alpha|H|)
        Use row-wise pgd
        '''

        X = self.X
        W = self.W
        alpha = 0 if self.alpha is None else self.alpha

        A = W.T @ W
        B = W.T @ X

        H1 = H0.copy()

        i = 0
        dist = np.inf
        while (i < sub_iter) and (dist > stopping_diff):
            H1_old = H1.copy()
            for k in np.arange(H1.shape[0]):
                grad = np.dot(A[k, :], H1) - B[k, :] + alpha * np.ones(H1.shape[1])
                H1[k, :] = H1[k, :] - (1 / (((i + 1) ** 0) * (A[k, k] + 1))) * eta*grad
                # nonnegativity constraint
                H1[k, :] = np.maximum(H1[k, :], np.zeros(shape=(H1.shape[1],)))
                #H0 = H1

            dist = LA.norm(H1 - H1_old, 'fro') / LA.norm(H1_old, 'fro')
            i = i + 1
        return H1

    def step(self):
        raise NotImplementedError


class ONMF(NMFOptim):

    def __init__(self, W, data_rows=28, **kwargs):
        super().__init__(W, **kwargs)
        self.step_count = 1
        self.data_rows = data_rows
        self.aggregates = np.zeros((self.n_components, self.n_components)), np.zeros((self.data_rows, self.n_components))

    def code_and_update_aggregates(self, sub_iter=10):
        A, B = self.aggregates
        X, H0 = self.X, self.H

        H = self.update_code(H0, sub_iter=sub_iter)

        n = self.step_count

        A = (n-1)/n*A + (1/n)*np.dot(H, H.T)
        B = (n-1)/n*B + (1/n)*np.dot(X, H.T)
        self.H = H
        self.aggregates = A, B
        self.step_count += 1

    def update_dict(self, sub_iter=10, stopping_diff=0.01):
        A, B = self.aggregates
        k = self.n_components
        W1 = self.W.copy()
        d, r = np.shape(W1)
        i=0
        dist = np.inf

        while (i < sub_iter) and (dist > stopping_diff):
            W1_old = W1.copy()
            for j in range(k):
                step_1 = 1/(A[j, j] + 1)
                u = step_1*(B[:, j] - np.dot(W1, A[:, j])) + W1[:, j]
                step_2 = 1/max(LA.norm(u), 1)
                W1[:, j] = step_2*u
                W1[:, j] = np.maximum(W1[:, j], np.zeros(shape=(d,)))
                W1[:, j] = (1 / np.maximum(1, LA.norm(W1[:, j]))) * W1[:, j]
            dist = LA.norm(W1 - W1_old, 'fro') / LA.norm(W1_old, 'fro')

        self.W = W1

    def step(self):
        self.code_and_update_aggregates()
        self.update_dict()


class AdaGrad(NMFOptim):

    def __init__(self, W, lr=1e-3, eps=1e-10, **kwargs):
        super().__init__(W, **kwargs)
        self.lr = lr
        self.eps = eps
        self.W_sum = None
        self.H_sum = None

    def step(self):
        A = self.W.T @ self.W
        B = self.W.T @ self.X

        H1 = self.H.copy()

        if self.H_sum is None:
            self.H_sum = {}

        # this is different
        if self.curr_node not in self.H_sum:
            self.H_sum[self.curr_node] = np.zeros_like(self.H)

        H_sum = self.H_sum[self.curr_node]
        for k in np.arange(self.H.shape[0]):
            grad = np.dot(A[k, :], self.H) - B[k, :] + self.alpha*np.ones(self.H.shape[1])
            H_sum[k, :] = H_sum[k, :] + grad**2
            step_size = self.lr*(1/(H_sum[k, :]**(1/2) + self.eps))
            H1[k, :] = self.H[k, :] - step_size * grad
            H1[k, :] = np.maximum(H1[k, :], np.zeros(shape=(H1.shape[1],)))

        self.H_sum[self.curr_node] = H_sum

        A = self.H @ self.H.T
        B = self.H @ self.X.T

        W1 = self.W.copy()
        d, r = np.shape(W1)

        if self.W_sum is None:
            self.W_sum = np.zeros_like(self.W)

        for k in np.arange(self.W.shape[1]):
            grad = np.dot(self.W, A[:, k]) - B.T[:, k]
            self.W_sum[:, k] = self.W_sum[:, k] + grad**2
            step_size = self.lr*(1/(self.W_sum[:, k]**(1/2) + self.eps))
            W1[:, k] = self.W[:, k] - step_size*grad
            W1[:, k] = np.maximum(W1[:, k], np.zeros(shape=(d,)))
            W1[:, k] = (1 / np.maximum(1, LA.norm(W1[:, k]))) * W1[:, k]

        self.H = H1
        self.W = W1


class PSGD(NMFOptim):

    def __init__(self, W, lr=1e-3, eta=0.5, **kwargs):
        super().__init__(W, **kwargs)
        self.step_count = 1
        self.lr = lr
        self.eta = eta

    def step(self):
        step_size = self.lr*float(self.step_count)**(-self.eta)

        A = self.W.T @ self.W
        B = self.W.T @ self.X

        H1 = self.H.copy()

        for k in np.arange(self.H.shape[0]):
            grad = np.dot(A[k, :], self.H) - B[k, :] + self.alpha * np.ones(self.H.shape[1])
            H1[k, :] = self.H[k, :] - step_size * grad
            H1[k, :] = np.maximum(H1[k, :], np.zeros(shape=(H1.shape[1],)))

        A = self.H @ self.H.T
        B = self.H @ self.X.T

        W1 = self.W.copy()
        d, r = np.shape(W1)

        for k in np.arange(self.W.shape[1]):
            grad = np.dot(self.W, A[:, k]) - B.T[:, k]
            W1[:, k] = self.W[:, k] - step_size * grad
            W1[:, k] = np.maximum(W1[:, k], np.zeros(shape=(d,)))
            W1[:, k] = (1 / np.maximum(1, LA.norm(W1[:, k]))) * W1[:, k]

        self.H = H1
        self.W = W1

        self.step_count += 1


class MU(NMFOptim):

    def __init__(self, W, rho=0, delta=0, eps=1e-5, **kwargs):

        super().__init__(W, **kwargs)
        self.rho = rho
        self.delta = delta
        self.eps = eps

    def step(self):
        H0 = self.H.copy()

        r = self.W.shape[1]
        I = np.eye(r)

        H1 = np.maximum(self.H, self.delta*np.ones_like(self.H))

        H0 = H1 * (np.dot(self.W.T, self.X) + self.rho*H1)
        H0 = H0 / ((np.dot(self.W.T, self.W) + self.rho*I) @ H1 + self.eps*np.ones_like(H0))

        #if np.isnan(H0).any():
            #H0 = self.H

        W1 = np.maximum(self.W, self.delta*np.ones_like(self.W))

        Wt = W1.T * (np.dot(self.H, self.X.T) + self.rho*W1.T)
        Wt = Wt/((np.dot(self.H, self.H.T) + self.rho*I) @ W1.T + self.eps*np.ones_like(Wt))

        #if np.isnan(Wt).any():
            #Wt = self.W.T

        self.H = H0
        self.W = Wt.T


class Rmiso(NMFOptim):

    def __init__(self, W, n_nodes=100, n_components=100, alpha=0, rho=0, beta=1, dynamic_reg=False, dr=False):

        '''
        :param W: dictionary
        :param n_components: number of columns in dictionary matrix W
        :param iterations: number of iterations where each iteration is a call to step()
        '''

        self.rho = rho
        self.beta = beta
        self.dr = dr
        self.dynamic_reg = dynamic_reg
        self.aggregates = None
        # dictionary to store the past code matrix H for each node
        self.code_mats = {}
        self.step_count = 1
        self.return_times = np.zeros(n_nodes)

        super().__init__(W, n_nodes=n_nodes, n_components=n_components, alpha=alpha)

    def init_surrogate(self):
        H0 = self.H
        self.H = self.update_code(H0)
        self.code_mats[self.curr_node] = self.H.copy()
        n = self.n_nodes
        if self.aggregates is not None:
            A, B = self.aggregates
            A = A + self.H @ self.H.T/n
            B = B + self.H @ self.X.T/n
        else:
            A = self.H @ self.H.T/n
            B = self.H @ self.X.T/n
        self.aggregates = (A, B)

    def code_and_update_aggregates(self, sub_iter=10, stopping_diff=0.1):
        A, B = self.aggregates
        X, H_old = self.X, self.code_mats[self.curr_node]

        H = self.update_code(H_old, sub_iter=sub_iter, stopping_diff=stopping_diff)
        n = self.n_nodes

        A = A + (H @ H.T - H_old @ H_old.T)/n
        B = B + (H - H_old) @ X.T/n

        self.code_mats[self.curr_node] = H.copy()

        self.H = H
        self.aggregates = A, B

    def update_dict(self, stopping_diff=0.01, sub_iter=1, rad=None):
        W = self.W
        A, B = self.aggregates
        d, r = np.shape(W)
        W1 = W.copy()
        i = 0
        dist = np.inf

        if self.dynamic_reg:
            self.return_times = self.return_times + np.ones(self.n_nodes)
            self.return_times[self.curr_node] = 0
            self.rho = self.beta*np.amax(self.return_times)

        rho = self.rho if not self.dr else 0

        while (i < sub_iter) and (dist > stopping_diff):
            W1_old = W1.copy()
            for j in np.arange(r):
                grad = np.dot(W1, A[:, j]) - B.T[:, j] + rho*(W1[:, j] - W[:, j])
                step_size = (1 / (A[j, j] + 1 + rho))
                W1[:, j] = W1[:, j] - step_size * grad
                W1[:, j] = np.maximum(W1[:, j], np.zeros(shape=(d, )))
                W1[:, j] = (1/np.maximum(1, LA.norm(W1[:, j])))*W1[:, j]

            if rad is not None:
                W1 = self.project_onto_intersection(W1, W, rad)

            dist = LA.norm(W1 - W1_old, 'fro')/LA.norm(W1_old, 'fro')
            i = i+1

        self.W = W1

    def step(self):
        r = float(self.step_count)**(-0.5)*np.log(float(self.step_count+1))**(-0.501) if self.dr else None
        self.code_and_update_aggregates(sub_iter=10)
        self.update_dict(sub_iter=10, rad=r)
        self.step_count += 1

    @staticmethod
    def project_onto_intersection(W, W0, rad):
        d, r = np.shape(W)
        dist = LA.norm(W - W0, 'fro')

        stopping_diff = dist
        i = 0

        while i < 10 and stopping_diff > 0.001:
            alpha = max(0, 1 - rad / dist)
            W_old = W.copy()
            W = W - alpha * (W - W0)
            for j in np.arange(r):
                W[:, j] = np.maximum(W[:, j], np.zeros(shape=(d,)))
            for j in np.arange(r):
                W[:, j] = (1 / np.maximum(1, LA.norm(W[:, j]))) * W[:, j]

            dist = LA.norm(W - W0, 'fro')
            stopping_diff = LA.norm(W - W_old, 'fro')

        return W




