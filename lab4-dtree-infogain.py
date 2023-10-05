import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

df = pd.DataFrame()
try:
    df = pd.read_csv("tennis.csv")
except FileNotFoundError:
    print("File not found. Please ensure the file path is correct.")

# Encode categorical variables
le = LabelEncoder()
df_encoded = df.apply(le.fit_transform)

# Split data into features and target variable
X = df_encoded[['Outlook', 'Temperature', 'Humidity', 'Windy']]
y = df_encoded['Play']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and fit the model
clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

# Plot the tree
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.show()

# Functions to calculate entropy and information gain
def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

def InfoGain(data,split_attribute_name,target_name="Play"):
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain

# Print Information Gain for each attribute
print("Info Gain of 'Outlook' is:", InfoGain(df, "Outlook"))
print("Info Gain of 'Temperature' is:", InfoGain(df, "Temperature"))
print("Info Gain of 'Humidity' is:", InfoGain(df, "Humidity"))
print("Info Gain of 'Windy' is:", InfoGain(df, "Windy"))
