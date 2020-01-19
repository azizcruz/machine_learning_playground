import tensorflow
import keras
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

data = pd.read_csv('student-mat.csv', sep=";") # we read the csv file first
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]] # we get the columns that we need and they need to be numbers

predict = "G3" # we want to predict final grade using the columns up

x = np.array(data.drop([predict], 1)) # we take all the chosen columns and put them in one array except the column we want to study
y = np.array(data[predict]) # the column we want to study a separated array

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # we get out the train results from both models + the test train so we can get the acc with it, beware of the order

best = 0
# for _ in range(30): # we start training until we get the best acc
#     x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
#     linear = linear_model.LinearRegression()
#     linear.fit(x_train, y_train)
#     acc = linear.score(x_test, y_test)
#     print(int(acc * 100) )
#
#     if acc > best: # we store the best acc in a pickel
#         best = acc
#         # Save the model for later use
#         with open('studentmodel.pickel', 'wb') as f:
#             pickle.dump(linear, f)

pickle_in = open('studentmodel.pickel', 'rb')
linear = pickle.load(pickle_in)

print("Co: \n", linear.coef_)
print("Interception: \n", linear.intercept_)
print("============Prediction==============")
prediction = linear.predict(x_test)

for x in range(len(prediction)): # here we compare the predictions with the actual nums
    print(int(prediction[x]), x_test[x], y_test[x]) # last column is the actual num

# Using matplotlib to show graph of the model result
p = 'G1' # X value
style.use('ggplot')
pyplot.scatter(data[p], data['G3']) # set x and y
pyplot.xlabel(p) # set the labels on the graph
pyplot.ylabel('Final Grade')
pyplot.show()
