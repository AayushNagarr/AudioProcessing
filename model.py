from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


# Load the dataset from the CSV file
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
data = pd.read_csv("train.csv")
val_data = pd.read_csv("test.csv")
id = val_data['id']

X = data.drop(columns=["filename", "length", 'label'])  # Replace "target_column_name" with the name of your target column
X = X.drop(columns=[col for col in X.columns if col.endswith('_var')])
Xscaler = StandardScaler()
Xscaler.fit(X)
X = Xscaler.transform(X)

mean_of_means = X.filter(like='mean', regex=True).mean(axis=1)
var_of_means = X.filter(like='mean', regex=True).var(axis=1)
X_m = pd.DataFrame()
X_m['mom'] = mean_of_means
X_m['vom'] = var_of_means
y = data["label"]

X_val = val_data.drop(columns=[ "length", "id"])  # Replace "target_column_name" with the name of your target column
X_val = X_val.drop(columns=[col for col in X_val.columns if col.endswith('_var')])

X_train, X_test, y_train, y_test = train_test_split(X_m, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

rf_classifier = RandomForestClassifier(n_estimators=500, random_state=42)  # You can adjust the number of trees (n_estimators)
rf_classifier.fit(X_train_scaled, y_train)


# Make predictions on the test data

 
y_pred = rf_classifier.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


y_pred = rf_classifier.predict(X_val_scaled)

y_pred_index = []
for i in y_pred:
    y_pred_index.append(genres[i])
predictions = np.array(y_pred_index)
submission_df = pd.DataFrame({'id': id,  'label': predictions})
submission_df.to_csv('music_genre_predictions.csv', index=False)
