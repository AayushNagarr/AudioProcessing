from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

class AudioKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.knn_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.preprocess_data(X_test)
        return self.knn_classifier.predict(X_test)

# Example usage
# Assuming your data is loaded into a DataFrame called 'df'
data = pd.read_csv('./train.csv')
X = data.drop(['filename', 'length', 'label'], axis = 1)
y = data['label']
print(data.head(10), X.head(10), y.head(10))
# Drop 'filename' and 'label' columns to get the feature matrix X
# X = df.drop(['filename', 'label'], axis=1)


# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the AudioKNNClassifier
# audio_knn = AudioKNNClassifier(n_neighbors=5)
# audio_knn.train(X_train, y_train)

