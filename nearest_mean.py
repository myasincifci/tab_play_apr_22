from data import DS
import numpy as np

class NMClassifier():

    def __init__(self) -> None:
        pass 
        
    def fit(self, X, t):

        t = t.reshape(t.shape[0])
        
        # zeros
        X_zeros = X[t == 0]
        t_zeros = t[t == 0]

        self.mean_0 = X_zeros.mean(axis=0)

        # ones
        X_ones = X[t == 1]
        t_ones = t[t == 1]

        self.mean_1 = X_ones.mean(axis=0)

    def predict(self, X) -> None:
        
        dist_to_zero = (np.linalg.norm(self.mean_0 - X, axis=1)**2).sum(axis=1)
        dist_to_one  = (np.linalg.norm(self.mean_1 - X, axis=1)**2).sum(axis=1)

        return dist_to_zero > dist_to_one

if __name__ == '__main__':

    data = DS()
    X = data.X
    t = data.t

    print(X.shape)

    clf = NMClassifier()
    clf.fit(data.X, data.t)

    pred = clf.predict( np.random.random((1, 60, 13)) )

    pred = clf.predict( X ).astype(int)

    a = (pred == t.reshape(t.shape[0]) ).astype(int).sum()

    print(f'Train accuracy: { a / t.shape[0] }')