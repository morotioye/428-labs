import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# Sample data
data = pd.DataFrame({
    'Hours_Studied': [3.745401188, 9.507143064, 7.319939418, 5.986584842],
    'Review_Session': [0, 1, 0, 1],
    'Results': [0, 1, 1, 1]
})

# Step 1: Scatter plot visualization
fig, ax = plt.subplots()
colors = {0: 'red', 1: 'blue'}
ax.scatter(data['Hours_Studied'], data['Review_Session'], c=data['Results'].apply(lambda x: colors[x]), alpha=0.5)
plt.title('Scatter plot of the data')
plt.xlabel('Hours_Studied')
plt.ylabel('Review_Session')
plt.show()

# Step 2: Fit a logistic regression model
X = data[['Hours_Studied', 'Review_Session']]
y = data['Results']
model = LogisticRegression()
model.fit(X, y)

# Step 3: Output model coefficients and performance metrics
print(f"Intercept: {model.intercept_[0]}")
print(f"Coefficient for 'Hours_Studied': {model.coef_[0][0]}")
print(f"Coefficient for 'Review_Session': {model.coef_[0][1]}")

y_pred_prob = model.predict_proba(X)[:, 1]
y_pred = model.predict(X)
accuracy = accuracy_score(y, y_pred)
print(f"Accuracy: {accuracy:.4f}")

auc = roc_auc_score(y, y_pred_prob)
print(f"AUC: {auc:.4f}")

fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
