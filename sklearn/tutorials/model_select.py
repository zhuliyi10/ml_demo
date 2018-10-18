import numpy as np
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

digits = datasets.load_digits()
X = digits.data
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# 逻辑回归
def logistic_regression():
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print('LogisticRegression:', clf.score(X_test, y_test))


# 支持向量机
def svc():
    clf = SVC(kernel='linear')
    clf.fit(X_train, y_train)
    print('SVC:', clf.score(X_test, y_test))


# k 近邻
def knn():
    clf = KNeighborsClassifier()
    # knn.fit(X_train, y_train)
    # print('knn:', knn.score(X_test, y_test))
    accuracy = cross_val_score(clf, X, y, cv=10, scoring='accuracy')  # 分cv 组进行交叉测试
    print('KNN', np.mean(accuracy))


# 梯度下降法
def sgd():
    clf = SGDClassifier()
    clf.fit(X_train, y_train)
    print('SGD:', clf.score(X_test, y_test))


# 决策树
def decision_tree():
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    print('DecisionTree:', clf.score(X_test, y_test))


# 高斯朴素贝叶斯
def gaussian_nb():
    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print('GaussianNB:', clf.score(X_test, y_test))


# 神经网络
def network():
    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    print('MLPClassifier:', clf.score(X_test, y_test))


if __name__ == '__main__':
    logistic_regression()
    svc()
    knn()
    sgd()
    decision_tree()
    gaussian_nb()
    network()
