import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Your Data
data = {
    'win': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    'click': [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    'team': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'claim': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    'reminder': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'Category': ['Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']
}
df = pd.DataFrame(data)

# Features and Labels
X = df.drop('Category', axis=1)
y = df['Category']

# Splitting Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# User Input: k Value
k = int(input("Please enter the number of neighbors (k value): "))

# KNN Model
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Classification Report
print(classification_report(y_test, predictions))
