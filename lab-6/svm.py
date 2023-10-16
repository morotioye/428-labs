import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, KFold

# Load data from "student_data.csv"
data = pd.read_csv("/Users/wnr/Documents/umbc/is428/labs/lab6/student_data.csv")

X = data[['Hours_Studied', 'Review_Session']]
y = data['Results']

# (1) SVM with linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X, y)
print("SVM with Linear Kernel fitted successfully!")

# (2) SVM with RBF kernel & Grid Search for best gamma
parameters = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100]}
svm_rbf = SVC(kernel='rbf')
grid_search = GridSearchCV(svm_rbf, parameters, cv=KFold(n_splits=2))  # Using 2 folds for k-fold cross-validation due to small dataset
grid_search.fit(X, y)

print(f"Best gamma for RBF kernel: {grid_search.best_params_['gamma']}")
print("SVM with RBF Kernel (with best gamma) fitted successfully!")
