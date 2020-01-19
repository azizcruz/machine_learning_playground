import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

df = pd.read_csv('car.data')

# We need to transform data inside the data frame to numbers, and we use preprocessing from sklearn to to do this.
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(df['buying']))
maint = le.fit_transform(list(df['maint']))
door = le.fit_transform(list(df['door']))
persons = le.fit_transform(list(df['persons']))
lug_boot = le.fit_transform(list(df['lug_boot']))
safety = le.fit_transform(list(df['safety']))
clas = le.fit_transform(list(df['class']))

predict = 'class'

x = list(zip(buying, maint, door, persons, lug_boot, safety)) # features
y = list(clas) # labels 'predict'

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1) # begin the action :D

