import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=4)
print(y_train)
print(y_test)

n_range = range(1, 31)
n_scores = []
for k in n_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # knn.fit(x_train,y_train)
    # scores=knn.score(x_test,y_test)
    # n_scores.append(scores)
    scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')
    n_scores.append(scores.mean())
plt.plot(n_range, n_scores)
plt.xlabel('value of k for KNN')
plt.ylabel('cross_validated accuracy')
plt.show()
