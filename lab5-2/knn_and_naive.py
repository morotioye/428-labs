import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load data
df = pd.read_csv('/Users/wnr/Documents/umbc/is428/labs/lab5-2/spam_dataset.csv')

# Split data into training and testing sets
X = df[['word1', 'word2', 'word3', 'word4']]
y = df['spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

nb_predictions = nb.predict(X_test)

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

knn_predictions = knn.predict(X_test)

# Evaluate models
print("Naive Bayes")
print("Accuracy:", accuracy_score(y_test, nb_predictions))
print("Classification Report:")
print(classification_report(y_test, nb_predictions))

print("\nK-Nearest Neighbors")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print("Classification Report:")
print(classification_report(y_test, knn_predictions))
