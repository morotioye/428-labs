import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Load data
df = pd.read_csv('predictions.csv')

# Assuming a threshold of 0.5; modify as per your assignment
threshold = 0.5

# Calculate binary prediction based on threshold
df['binary_prediction'] = (df['Prediction'] >= threshold).astype(int)

# Step 1: Create Confusion Matrix
cm = confusion_matrix(df['True_Label'], df['binary_prediction'])
TN, FP, FN, TP = cm.ravel()

# Step 2: Calculate TPR and FPR
TPR = TP / (TP + FN)  # True Positive Rate (Sensitivity)
FPR = FP / (FP + TN)  # False Positive Rate (1 - Specificity)

print(f'True Positive Rate: {TPR:.2f}')
print(f'False Positive Rate: {FPR:.2f}')

# Step 3: Plot ROC Curve
fpr, tpr, _ = roc_curve(df['True_Label'], df['Prediction'])
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.scatter([FPR], [TPR], color='red')  # Plotting the calculated point
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
