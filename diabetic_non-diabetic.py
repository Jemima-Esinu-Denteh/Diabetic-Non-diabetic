import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.display import display
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics # Import scikit-learn metrics module for accuracy calculation
col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age', 'label']
data = pd.read_csv(r"C:\Users\jemmy\Documents\Skills Bootcamp\Edgehill University\Excel Files\diabetes.csv", header=None, names=col_names)
print(data.head())

# split dataset in features and target variable
feature_cols = ['glucose', 'bmi', 'age', 'insulin', 'pregnant']
X = data[feature_cols] # Features
y = data.label # Target variable

# Split dataset into training set and test set with 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Random Forest with default parameters
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
print('Training Accuracy: {:.2f}'.format(rf.score(X_train, y_train)))
print('Test Accuracy: {:.2f}'.format(rf.score(X_test, y_test)))

# Tuning to find the best parameters
rf_tuned = RandomForestClassifier()
rf_tuned.fit(X_train, y_train)
y_pred = rf_tuned.predict(X_test)
print('Test Accuracy: {:.3f}'.format(metrics.accuracy_score(y_pred, y_test)))
cv_scores = cross_val_score(rf_tuned, X, y, cv=5)

# Plot predicted and the actual labels over time for the testing samples
plt.figure()
y_test=y_test.to_numpy()
plt.plot(y_test,label="Actual")
plt.plot(y_pred,label="Predicted")
plt.tick_params(labelsize=16)
plt.legend(loc='best', prop={'size': 20})
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.text(250, 1.1, r'1-',fontsize=20)
plt.legend(loc='best', prop={'size': 20})
plt.ylabel('1: Diabetic, 2-Non-Diabetic', fontsize=16)
plt.xlabel('Number of Patients', fontsize=16)
plt.show()

# Parameter search for random forest
param_dist = {'n_estimators': [50,100,150,200,250],
               "max_features": [0,1,2,3,4,5,6,7,8,9],
               'max_depth': [1,2,3,4,5,6,7,8,9],
               "criterion": ['gini','entropy']}
rf = RandomForestClassifier()
rf_cv = RandomizedSearchCV(rf, param_distributions = param_dist, cv =5, random_state=0, n_jobs = 1)
rf_cv.fit(X_train,y_train)
print("Tuned Random Forest Parameters: {}".format(rf_cv.best_params_))
print('Best cross-validation score: {:.3f}'.format(rf_cv.best_score_))

results = pd.DataFrame(rf_cv.cv_results_)
display(results.head())
rf_param = RandomForestClassifier(n_estimators=250, max_features=1, max_depth=9, criterion='gini', random_state = 44)
rf_param.fit(X_train, y_train)
print('Training Accuracy: {:.2f}'.format(rf.score(X_train, y_train)))
print('Test Accuracy: {:.2f}'.format(rf.score(X_test, y_test)))