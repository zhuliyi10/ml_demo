import pickle

from sklearn import datasets
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split


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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    # saveJoblibModel(x_train,y_train)
    clf = restoreJoblibModel()
    print(clf.score(x_test, y_test))
    print(clf.predict(x_test))
    print(y_test)
