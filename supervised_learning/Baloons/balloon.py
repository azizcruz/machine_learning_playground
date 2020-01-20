import sklearn
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn import preprocessing
import pickle

df = pd.read_csv('adult+stretch.data')

le = preprocessing.LabelEncoder()
color = le.fit_transform(list(df['color']))
size = le.fit_transform(list(df['size']))
act = le.fit_transform(list(df['act']))
age = le.fit_transform(list(df['age']))
inflated = le.fit_transform(list(df['inflated']))

x = list(zip(color, size, act, age, inflated))
y = list(age)

names = ['ADULT', 'CHILD']
predict = 'age'

best = 0.95
for _ in range(30):
    x_train, x_sample, y_train, y_sample = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    model = KNeighborsClassifier(n_neighbors=9)
    model.fit(x_train, y_train)
    acc = model.score(x_train, y_train)

    if acc > best:
        best = acc
        # Save the model for later use
        with open('age_classify.pickel', 'wb') as f:
            pickle.dump(model, f)

predicted = model.predict(x_train)
for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], x_train[x], 'Actual: ',names[y_train[x]])
