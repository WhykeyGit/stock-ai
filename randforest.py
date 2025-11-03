from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample

'''
Data to predict when using random forest:
- Stock price movement (up/down) based on historical data
Most common prediction is the 50 days moving average and the 200 days moving average
'''


class RandForest:
    def __init__(self, n_trees=100, max_depth=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            X_sample, y_sample = resample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        import numpy as np
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.squeeze(np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=tree_preds))