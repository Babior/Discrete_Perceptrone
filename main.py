import matplotlib.pyplot as plt
import pandas as pd
from perceptron import Perceptron


def load_data(filename):
    """
    Function to load txt file into a dict
        :param str filename : string containing name of the file
        :return dict data_dict : dictionary containing array of decision attributes and target array
    """
    df_iris_train = pd.read_csv(filename, header=None, sep='\s+', decimal=',')

    data_array = df_iris_train.iloc[:, :-1].to_numpy()
    target_array = df_iris_train.iloc[:, -1].to_numpy()
    data_dict = {
        "data": data_array,
        "target": target_array
    }
    return data_dict


iris_train = load_data('iris_training.txt')
X = iris_train["data"]
y = pd.Series(iris_train["target"], dtype="category").cat.codes.values
plt.title('Dataset')
plt.scatter(X[:40, 0], X[:40, 1], color='green', marker='x', label='setosa')
plt.scatter(X[40:80, 0], X[40:80, 1], color='red', marker='o', label='versicolor')
plt.scatter(X[80:120, 0], X[80:120, 1], color='blue', marker='o', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper right')
plt.show()
plt.scatter(X[:40, 2], X[:40, 3], color='green', marker='x', label='setosa')
plt.scatter(X[40:80, 2], X[40:80, 3], color='red', marker='o', label='versicolor')
plt.scatter(X[80:120, 2], X[80:120, 3], color='blue', marker='o', label='virginica')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.legend(loc='upper right')
plt.show()

per = Perceptron(learning_rate=0.1, n_iter=100, random_state=1)
per.fit(X, y)
plt.plot(range(1, len(per.errors_) + 1), per.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()
