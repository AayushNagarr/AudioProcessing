import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt


from sklearn.metrics import accuracy_score

import pandas as pd


genres = {
    'blues':0,
    'classical':1,
    'country':2,
    'disco':3,
    'hiphop':4,
    'jazz':5,
    'metal':6,
    'pop':7,
    'reggae':8,
    'rock':9
}
def convertlabel(df):
    def convert(x):
        return genres.get(x)
    df['label'] = df.apply(lambda row: convert(row['label']), axis = 1)
    return df['label']

def preprocess(filename):
    df = pd.read_csv(filename)
    X = df.drop(['filename', 'length', 'label'], axis = 1)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    y = convertlabel(df)
    return X,y
    
X,y = preprocess('./train.csv')
# EDA

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the BaggingClassifier
param_grid = {
    'C': [1,10, 100],
    'gamma': ['auto',6, 11]
}
svm = SVC()
# model = Pipeline(steps=[('standardscaler', StandardScaler()),('svc', SVC(kernel = 'rbf', C = 6.0, gamma = 'auto'))])

model = GridSearchCV(svm, param_grid, cv=5)

model.fit(X_train, y_train)
model = model.best_estimator_

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# TEST

testdf = pd.read_csv('./test.csv')
id = testdf['id']
X_val = testdf.drop(['length', 'id'], axis = 1)
scaler = StandardScaler()

X_val = scaler.fit_transform(X_val)


test = pd.read_csv('./final_test.csv')
y_val = model.predict(X_val)
print(accuracy_score(y_val, test['label']))
submission = pd.DataFrame({'id': id, 'label' : y_val})
submission.to_csv('final.csv', index = False)







# y_pred_index = []
# for i in y_pred:
#     y_pred_index.append(i)
# predictions = np.array(y_pred_index)
# submission_df = pd.DataFrame({'id': id,  'label': predictions})
# submission_df.to_csv('music_genre_predictions.csv', index=False)