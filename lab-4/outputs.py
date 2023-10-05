from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

true_labels = [1, 0, 1, 0, 1, 1, 0, 0, 1, 0]
predicted_labels = [1, 1, 1, 0, 1, 0, 0, 1, 1, 0]

# Confusion Matrix
tn, fp, fn, tp = confusion_matrix(true_labels, predicted_labels).ravel()

# Performance Metrics
accuracy = accuracy_score(true_labels, predicted_labels)
precision = precision_score(true_labels, predicted_labels)
recall = recall_score(true_labels, predicted_labels)
fpr = fp / (fp + tn)  # False Positive Rate

# Displaying the results
print(f"Confusion Matrix:\nTP={tp}, FP={fp}, FN={fn}, TN={tn}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"False Positive Rate: {fpr:.2f}")
