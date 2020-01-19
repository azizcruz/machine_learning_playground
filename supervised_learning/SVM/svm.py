#
#   Support Victor Machines
#

import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import  KNeighborsClassifier

cancer = datasets.load_breast_cancer()

x = cancer.data # get the data and store it in x, ofcourse without the target column
y = cancer.target # get the target column data

# here we set up training
x_train, x_sample, y_train, y_sample = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ['malignant' 'benign'] # the classes that we changed them to numbers 0, 1

clf = svm.SVC(kernel='linear', degree=2)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_sample) # predict y

acc = metrics.accuracy_score(y_sample, y_pred) # compare the y sample with the prdicted y

print(acc)


