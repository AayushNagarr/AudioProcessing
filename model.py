from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

class AudioKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.scaler = StandardScaler()

    def preprocess_data(self, X):
        # Standardize the features
        return self.scaler.transform(X)

    def train(self, X_train, y_train):
        X_train = self.preprocess_data(X_train)
        self.knn_classifier.fit(X_train, y_train)

    def predict(self, X_test):
        X_test = self.preprocess_data(X_test)
        return self.knn_classifier.predict(X_test)

    def evaluate(self, y_true, y_pred):
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred)
        print(f'Accuracy: {accuracy}')
        print('Classification Report:\n', report)

# Example usage
# Assuming your data is loaded into a DataFrame called 'df'
# Drop 'filename' and 'label' columns to get the feature matrix X
X = df.drop(['filename', 'label'], axis=1)

# Extract labels into a separate variable y
y = df['label']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the AudioKNNClassifier
audio_knn = AudioKNNClassifier(n_neighbors=5)
audio_knn.train(X_train, y_train)

# Make predictions on the test set
y_pred = audio_k
