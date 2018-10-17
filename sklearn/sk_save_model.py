import pickle

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib


def trainModel(x, y):
    clf = svm.SVC()
    clf.fit(x, y)
    return clf


# method 1 :pickle
def savePickleModel(x, y):
    clf = trainModel(x, y)
    with open('save/clf.pickle', 'wb')as f:
        pickle.dump(clf, f)


def restorePickleModel():
    with open('save/clf.pickle', 'rb')as f:
        return pickle.load(f)


#  method 2 : joblib
def saveJoblibModel(x, y):
    clf = trainModel(x, y)
    joblib.dump(clf, "save/clf.pkl")


def restoreJoblibModel():
    clf = joblib.load("save/clf.pkl")
    return clf


if __name__ == '__main__':
    iris = datasets.load_iris()
    x, y = iris.data, iris.target
    # saveJoblibModel(x,y)
    clf = restoreJoblibModel()
    p = x
    print(clf.predict(p))
    print(y)
