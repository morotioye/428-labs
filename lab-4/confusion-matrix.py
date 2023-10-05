import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Example true labels and predictions
true_labels = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0]
predictions = [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0]

# Calculate metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions)
recall = recall_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions)

# Create a figure and a set of subplots
fig, ax = plt.subplots(2, 1, figsize=(8, 10))

# Add text in the first subplot
ax[0].text(0.5, 0.8, f'Accuracy: {accuracy:.2f}', fontsize=12, ha='center')
ax[0].text(0.5, 0.6, f'Precision: {precision:.2f}', fontsize=12, ha='center')
ax[0].text(0.5, 0.4, f'Recall: {recall:.2f}', fontsize=12, ha='center')
ax[0].text(0.5, 0.2, f'F1 Score: {f1:.2f}', fontsize=12, ha='center')

# Hide axes for the first subplot
ax[0].axis('off')

# Plot confusion matrix in the second subplot
conf_matrix = confusion_matrix(true_labels, predictions)
cax = ax[1].matshow(conf_matrix, cmap=plt.cm.Blues)
fig.colorbar(cax)
ax[1].set_xlabel('Predicted')
ax[1].set_ylabel('True')
ax[1].xaxis.set_ticks_position('bottom')
ax[1].set_title('Confusion Matrix', pad=20)

# Add text on each cell in the confusion matrix
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax[1].text(j, i, str(conf_matrix[i, j]), va='center', ha='center')

# Save the figure
plt.savefig('evaluation_metrics_and_conf_matrix.png', bbox_inches='tight', dpi=300)

# Show the figure
plt.show()
