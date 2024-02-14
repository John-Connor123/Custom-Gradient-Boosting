import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings("ignore")


def load_numpy_dataset():
    # Load Boston house prices dataset
    df = pd.read_csv(
        filepath_or_buffer="http://lib.stat.cmu.edu/datasets/boston",
        delim_whitespace=True,
        skiprows=21,
        header=None,
    )
    columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
               'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV', ]

    # Flatten all the values into a single long list and remove the nulls
    values_w_nulls = df.values.flatten()
    all_values = values_w_nulls[~np.isnan(values_w_nulls)]

    # Reshape the values to have 14 columns and make a new df out of them
    X = pd.DataFrame(
        data=all_values.reshape(-1, len(columns)),
        columns=columns,
    )
    return np.array(X)


X = load_numpy_dataset()
y = X[:, -1]
X = X[:, :-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# 1) Implement.
class MyDecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.tree_ = {}
        self.max_depth_ = max_depth

    @staticmethod
    def mse(y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def fit(self, X, y, tree_path='0'):
        if len(tree_path) - 1 == self.max_depth_ or X.shape[0] <= 1:
            self.tree_[tree_path] = np.mean(y)
            return

        minimum_mse = None
        best_split = None

        # For every unique value of every feature of X split the data in two parts:
        #   one part are observations which are less than or equal to this value
        #   second part are the others
        for feature in range(X.shape[1]):
            for value in sorted(set(X[:, feature])):

                less_than_or_equal_obs = X[:, feature] <= value  # select observations
                # which are less than or equal to value

                # one part are observations which are less than or equal to this value
                X1, y1 = X[less_than_or_equal_obs], y[less_than_or_equal_obs]

                # second part are the others
                X2, y2 = X[~less_than_or_equal_obs], y[~less_than_or_equal_obs]

                # Calculate weighted MSE for a split
                MSE1 = self.mse(y1, np.mean(y1))
                MSE2 = self.mse(y2, np.mean(y2))
                weight_1 = len(y1) / len(y)
                weight_2 = len(y2) / len(y)
                weighted_mse = MSE1 * weight_1 + MSE2 * weight_2

                # Update MSE
                if minimum_mse is None or weighted_mse < minimum_mse:
                    minimum_mse = weighted_mse
                    best_split = (feature, value)

        # Get samples with best split
        feature, value = best_split
        splitting_condition = X[:, feature] <= value
        X1, y1, X2, y2 = X[splitting_condition], y[splitting_condition], \
                         X[~splitting_condition], y[~splitting_condition]

        #  Add the splitting condition to tree
        self.tree_[tree_path] = best_split

        # Continue growing the tree
        self.fit(X1, y1, tree_path=tree_path + '0')
        self.fit(X2, y2, tree_path=tree_path + '1')

    def predict(self, X):
        results = []
        for i in range(X.shape[0]):
            tree_path = '0'
            while True:
                value_for_path = self.tree_[tree_path]
                if type(value_for_path) != tuple:
                    result = value_for_path
                    break
                feature, value = value_for_path
                if X[i, feature] <= value:
                    tree_path += '0'
                else:
                    tree_path += '1'
            results.append(result)
        return np.array(results)


class MyGradientBoostingRegressor:
    def fit(self, X, y):
        self.trees = []  # Create a list to store trees

        for i in range(100):
            tree = MyDecisionTreeRegressor(3)
            tree.fit(X, y - self.predict(X))  # Fit the tree to data
            self.trees.append(tree)  # Add the tree to the list of trees

    def predict(self, X):
        # Create array to store predictions
        trees_predictions = np.zeros((len(X), len(self.trees)))

        # Predict for each observation for each tree
        for i, tree in enumerate(self.trees):
            # trees_predictions[:, i] - i-th column, converted by slice in 1d array
            trees_predictions[:, i] = tree.predict(X) * (1 if i == 0 else 0.1)

        return np.sum(trees_predictions, axis=1)


def mse(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))


# 2) Measure performance — sklearn.
np.random.seed(42)
sklearn_model = GradientBoostingRegressor()
sklearn_model.fit(X_train, y_train)
sklearn_y_pred = sklearn_model.predict(X_test)
print(f'Sklearn score:', mse(y_test, sklearn_y_pred))

# 3) Measure performance — our implementation.
np.random.seed(42)
my_model = MyGradientBoostingRegressor()
my_model.fit(X_train, y_train)
print(f'MyGradientBoostingRegressor score:', mse(y_test, my_model.predict(X_test)))
