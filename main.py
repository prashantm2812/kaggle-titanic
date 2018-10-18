import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
y_train = data['Survived']
data = data.append(pd.read_csv('test.csv'))
data = data.drop(['Survived', 'Name', 'Ticket', 'PassengerId', 'Cabin'], axis = 1)
data['FamMem'] = data['SibSp'] + data['Parch']
check = data.describe()
for i in list(check):
    data[i] = data[i].fillna(data[i].mean())
check_again = data.describe(include = 'all')
col = list(check_again)
for i in col :
    data[i] = data[i].fillna(data[i].value_counts().idxmax())
        
categorical = data.select_dtypes(exclude = ['number'])
columns_categorical = list(categorical)
        
col_ind = []
for i in columns_categorical:
    col_ind.append(data.columns.get_loc(i))
    
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for i in columns_categorical:
    labenc = LabelEncoder()
    data[i] = labenc.fit_transform(data[i])
ohenc = OneHotEncoder(categorical_features = col_ind)
data = ohenc.fit_transform(data).toarray()

X_train = data[0:891, :]
X_test = data[891:, :]

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
sc_test = StandardScaler()
X_test = sc_test.fit_transform(X_test)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 120, gamma = 0.012, random_state = 0)#79.904%
classifier.fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 20)
accuracies.mean()
accuracies.std()

C = np.arange(0.25, 200, 0.25)
gamma = np.arange(0.1, 10, 0.1)

from sklearn.model_selection import GridSearchCV
parameters = [{'C' : C, 'kernel' : ['rbf'], 'gamma' : gamma, 'random_state' : [0, 11, 42]}]
grid_search = GridSearchCV(estimator = classifier, param_grid = parameters, scoring = 'accuracy', cv = 20,
                           n_jobs = 1)
grid_search = grid_search.fit(X_train, y_train)
best_score = grid_search.best_score_
best_params = grid_search.best_params_

y_pred_test = classifier.predict(X_test)

y_pred1 = pd.DataFrame(y_pred_test)
y_pred1.to_csv('out3.csv', index = False, header = ['Survived'])
