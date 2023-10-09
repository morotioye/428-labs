# Import necessary libraries
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OneHotEncoder

# Sample dataset
data = {
    'Content': ["Click here to win a special prize!",
                "Team meeting rescheduled. Check the portal and click for details.",
                "Claim your free vacation package now!",
                "A reminder for tomorrow's team-building exercise.",
                "Win a brand new car by participating in our survey!",
                "The reminder for project deadlines is attached.",
                "Join our team and win exciting bonuses!",
                "Click on the link for the annual team outing details.",
                "Last chance to claim your lottery money!",
                "The team has shared the minutes of the meeting. Click to access."
                ],
    'win': [1, 0, 0, 0, 1, 0, 1, 0, 0, 0],
    'click': [1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
    'team': [0, 1, 0, 1, 0, 0, 1, 1, 0, 1],
    'claim': [0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    'reminder': [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
    'Category': ['Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham', 'Spam', 'Ham']
}

df_train = pd.DataFrame(data)

# Separating features and target variable
X_train = df_train[['win', 'click', 'team', 'claim', 'reminder']]
y_train = df_train['Category']

# Applying one-hot encoding
enc = OneHotEncoder()
X_train_encoded = enc.fit_transform(X_train).toarray()

# Train Naive Bayes Classifier
clf = MultinomialNB()
clf.fit(X_train_encoded, y_train)

# Sample test (just for demonstration, replace this with actual test data)
X_test_sample = [[1, 0, 0, 1, 0]]  # Example: "Click here to claim your prize!"
X_test_encoded_sample = enc.transform(X_test_sample).toarray()

# Predict the category
predicted_category = clf.predict(X_test_encoded_sample)
print(predicted_category)
