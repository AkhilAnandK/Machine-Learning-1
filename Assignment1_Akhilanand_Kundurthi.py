import sklearn
#Fetching Dataset from OpenML
from sklearn import datasets
sampleSet=datasets.fetch_openml(data_id=1489)
#Creating a decision tree object mytree
from sklearn import tree
mytree = tree.DecisionTreeClassifier(criterion="entropy")
#Training the data
mytree.fit(sampleSet.data,sampleSet.target)
#Making predictions
predictions = mytree.predict(sampleSet.data)
#Evaluation
from sklearn import metrics
metrics.accuracy_score(sampleSet.target, predictions)

#Decision tree with default parameters
print("\n-----------------------Start of Assignment 1-----------------------")
print("\nDecision tree with default parameters\n")
from sklearn import model_selection
dtc = tree.DecisionTreeClassifier()
cv = model_selection.cross_validate(dtc, sampleSet.data, sampleSet.target, scoring=["roc_auc"], cv=10)
print("AUC score measured for Decision tree with default parameters is: ",cv["test_roc_auc"].mean())

#Parameter Tuning
print("\nDecision tree with tuned min_samples_leaf using GridSearchCV\n")
parameters = [{"min_samples_leaf":[2,4,6,8]}]
dtc = tree.DecisionTreeClassifier()
tuned_dtc = model_selection.GridSearchCV(dtc, parameters, scoring="roc_auc", cv=10)
cv_tuned=model_selection.cross_validate(tuned_dtc, sampleSet.data, sampleSet.target, scoring=["roc_auc"], cv=10)
print("AUC score measured for Decision tree with tuned min_samples_leaf using GridSearchCV is: ",cv_tuned["test_roc_auc"].mean())

#Random Forest
print("\nRandom Forest\n")
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
cv_rf = model_selection.cross_validate(rf, sampleSet.data, sampleSet.target, scoring=["roc_auc"], cv=10)
print("AUC score measured for Random Forest is: ",cv_rf["test_roc_auc"].mean())

#Bagging
print("\nBagging\n")
from sklearn.ensemble import BaggingClassifier
bagged_dtc = BaggingClassifier()
cv_bagging=model_selection.cross_validate(bagged_dtc, sampleSet.data, sampleSet.target, scoring=["roc_auc"], cv=10)
print("AUC score measured for Bagging is: ",cv_bagging["test_roc_auc"].mean())

#AdaBoost
print("\nAdaBoost\n")
from sklearn.ensemble import AdaBoostClassifier
ada_dtc = AdaBoostClassifier()
cv_ada=model_selection.cross_validate(ada_dtc, sampleSet.data, sampleSet.target, scoring=["roc_auc"], cv=10)
print("AUC score measured for AdaBoost is: ",cv_ada["test_roc_auc"].mean())

print("\n-----------------------End of Assignment 1-----------------------")
