# -*- coding: utf-8 -*-
"""
Created on Sat May 09 19:12:44 2015

@author: Naman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the Training dataset file
df_ncaa = pd.read_csv("D:\Stevens_2nd Sem\Statistical Learning\NCAA - Project\Final_NCAA_Train.csv")
df_ncaa.head()
# Dropping the variables
df_ncaa_final = df_ncaa.drop(['Index','teamA','TeamName_x','scoreA','teamB','TeamName_y','scoreB','location'],axis=1)
df_ncaa_final.head()
# Dimensions of the training dataset
df_ncaa_final.shape
# Checking whether there are any Null values in the dataset. No Null values are there.
df_ncaa_final.isnull().sum()
# Splitting the dataset in the Training and Validation set. 
# 70% is the training set and 30% is the validation set.'df_ncaa_train' is the training set and 'df_ncaa_val' is the validation set.
from sklearn.cross_validation import train_test_split
df_ncaa_train, df_ncaa_val = train_test_split(df_ncaa_final,test_size = 0.30,random_state =222)
# Converting the datasets from numpy Dataframes.
df_ncaa_train = pd.DataFrame(df_ncaa_train,columns = df_ncaa_final.columns)
df_ncaa_val = pd.DataFrame(df_ncaa_val,columns = df_ncaa_final.columns)
df_ncaa_train.head()
# Dimensions of training and validation set
df_ncaa_train.shape , df_ncaa_val.shape
# Separating the features and target variables.
# features contain columns from 'fgm_diff' to 'de_oe_diff'. And target variable contain column 'winnner'. Categorical values of 'winner'
# A and B are converted to discrete values of 0 and 1.
features = df_ncaa_train.ix[:,'fgm_diff':'de_oe_diff']
target = df_ncaa_train[['winner']]
target['winner'] = target['winner'].apply(lambda x: 0 if x == 'A' else 1)
target.head()
# Dimensions of features and target set
features.shape, target.shape
# Separating the val_features and val_target variables.
# val_features contain columns from 'fgm_diff' to 'de_oe_diff'. And val_target variable contain column 'winnner'. Categorical values of 'winner'
# A and B are converted to discrete values of 0 and 1.
val_features = df_ncaa_val.ix[:,'fgm_diff':'de_oe_diff']
val_target = df_ncaa_val[['winner']]
val_target['winner'] = val_target['winner'].apply(lambda x: 0 if x == 'A' else 1)
val_target.head()
# Dimensions of val_features and val_target set
val_features.shape, val_target.shape

# ROC Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
def plot_roc_curve(actual, predicted):
     fpr, tpr, thresholds = roc_curve(actual,predicted)
     roc_auc = auc(fpr,tpr)
     plt.title('Receiver Operating Characteristic')
     plt.plot(fpr,tpr, 'b',
     label='AUC = %0.2f'% roc_auc)
     plt.legend(loc='lower right')
     plt.plot([0,1],[0,1],'r--')
     plt.xlim([-0.1,1.2])
     plt.ylim([-0.1,1.2])
     plt.ylabel('True Positive Rate')
     plt.xlabel('False Positive Rate')
     plt.show()
     return (fpr,tpr, roc_auc)

# Precision-Recall Curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
def plot_prre_curve(actual, predicted):
    precision, recall, threshold = precision_recall_curve(actual, predicted)   
    area = auc(recall, precision)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example: AUC=%0.2f' % area)
    plt.legend(loc="lower left")
    plt.show()
    
# Building various models on Training data and testing the results on the Validation data.
# Algorithms performed:-
# 1. Logistic Regression
# 2. Decision Tree Classifier(CART)
# 3. Naive Bayes
# 4. Support Vector Machine(SVM)
# 5. Random Forest Classifier
    
# Logistic Regression

import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")      # Ignore Warnings
# fit Logistic Regression model to the data
clf_log = LogisticRegression()
# fit a logistic regression model to the data
clf_log.fit(features,target)

# make predictions
preds_logreg = clf_log.predict(val_features)
print preds_logreg
probs = clf_log.predict_proba(val_features.astype(float))
print probs
#print pd.crosstab(test_target, preds_cart, rownames=['actual'], 
#            colnames=['prediction'])
# summarize the fit of the model
print "Classification Report:", (metrics.classification_report(val_target,preds_logreg))
print "Confusion Matrix: ", (metrics.confusion_matrix(val_target,preds_logreg))
print "Accuracy Score for Logistic Regression is :", metrics.accuracy_score(val_target,preds_logreg)
auc_logreg = roc_auc_score(val_target, preds_logreg)
print "AUC for Logistic Regression is:",auc_logreg
print "Test Error Rate: ",zero_one_loss(val_target, preds_logreg)
fpr_logreg,tpr_logreg,roc_auc_logreg = plot_roc_curve(val_target,probs[:,1])
print "False Positive Rate", fpr_logreg
print "True Positive Rate", tpr_logreg
print plot_prre_curve(val_target,probs[:,1])

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

# fit CART model to the data
clf_cart = DecisionTreeClassifier(criterion='entropy')
clf_cart.fit(features,target)
# make predictions
preds_cart = clf_cart.predict(val_features)
preds_cart
probs = clf_cart.predict_proba(val_features.astype(float))
print probs
print pd.crosstab(val_target, preds_cart, rownames=['actual'], 
            colnames=['prediction'])
# summarize the fit of the model
print "Classification Report:", (metrics.classification_report(val_target,preds_cart))
print "Confusion Matrix:", (metrics.confusion_matrix(val_target,preds_cart))
print "Accuracy Score for CART is :", metrics.accuracy_score(val_target,preds_cart)
auc_cart = roc_auc_score(val_target, preds_cart)
print "AUC for CART is:", auc_cart
print "Test Error Rate: ",zero_one_loss(val_target, preds_cart)
fpr_cart,tpr_cart,roc_auc_cart = plot_roc_curve(val_target,probs[:,1])
print "False Positive Rate", fpr_cart
print "True Positive Rate", tpr_cart
print plot_prre_curve(val_target,probs[:,1])

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score

gnb = GaussianNB()
# fit a Naive Bayes model to the data
gnb.fit(features,target)
# make predictions
preds_gnb = gnb.predict(val_features)
probs = gnb.predict_proba(val_features.astype(float))
print probs
# summarize the fit of the model
print "Classification Report: ", (metrics.classification_report(val_target,preds_gnb))
print "Confusion Matrix:", (metrics.confusion_matrix(val_target,preds_gnb))
print "Accuracy Score for Naive Bayes is:", metrics.accuracy_score(val_target,preds_gnb)
auc_gnb = roc_auc_score(val_target, preds_gnb)
print "AUC for Naive Bayes is:", auc_gnb
print "Test Error Rate: ",zero_one_loss(val_target, preds_gnb)
fpr_gnb,tpr_gnb,roc_auc_gnb = plot_roc_curve(val_target,probs[:,1])
print "False Positive Rate", fpr_gnb
print "True Positive Rate", tpr_gnb
print plot_prre_curve(val_target,probs[:,1])

# Support Vector Machine
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

# fit SVM model to the data
clf_svm = svm.SVC(probability=True,kernel='rbf')
clf_svm.fit(features,target)
# make predictions
preds_svm = clf_svm.predict(val_features)
probs = clf_svm.predict_proba(val_features.astype(float))
print probs
print preds_svm
# summarize the fit of the model
print "Classification Report: ", (metrics.classification_report(val_target,preds_svm))
print "Confusion Matrix: ", (metrics.confusion_matrix(val_target,preds_svm))
print "Accuracy Score:", metrics.accuracy_score(val_target,preds_svm)
auc_svm = roc_auc_score(val_target, preds_svm)
print "AUC for SVM is:",auc_svm
print "Test Error Rate: ",zero_one_loss(val_target, preds_svm)
fpr_svm,tpr_svm,roc_auc_svm = plot_roc_curve(val_target,probs[:,1])
print "False Positive Rate", fpr_svm
print "True Positive Rate", tpr_svm
print plot_prre_curve(val_target,probs[:,1])

# RandomForest Classification

from sklearn.ensemble import RandomForestClassifier
clf_rf = RandomForestClassifier(n_estimators = 100,random_state=42)
clf_rf = clf_rf.fit(features,target)
preds_rf = clf_rf.predict(val_features)
probs = clf_rf.predict_proba(val_features.astype(float))
probs
preds_rf
print "Classification Report: ", (metrics.classification_report(val_target,preds_rf))
print "Confusion Matrix: ", (metrics.confusion_matrix(val_target,preds_rf))
print "Accuracy Score:", metrics.accuracy_score(val_target,preds_rf)
auc_rf = roc_auc_score(val_target, preds_rf)
print "AUC for Random Forest is:", auc_rf
print "Test Error Rate: ",zero_one_loss(val_target, preds_rf)
fpr_rf,tpr_rf,roc_auc_rf = plot_roc_curve(val_target,probs[:,1])
print "False Positive Rate", fpr_rf
print "True Positive Rate", tpr_rf
print plot_prre_curve(val_target,probs[:,1])

importances = clf_rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(16):
    print("%d. feature %d (%f)" % (f+1 , indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(16), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(16), indices)
plt.xlim([-1, 16])
plt.show()
print "***************************"
print "Classification Score on Validation set:", clf_rf.score(val_features,val_target)

# Plot ROC curves 
plt.clf() 
plt.plot(fpr_logreg,tpr_logreg,fpr_cart, tpr_cart, fpr_gnb, tpr_gnb,fpr_svm, tpr_svm,fpr_rf,tpr_rf) 
print auc_logreg
plt.legend(["Logistic Regression (AUC=%0.2f)" % roc_auc_logreg,"Naive Bayes(AUC=%0.2f)" % roc_auc_gnb,"Random Forest(AUC=%0.2f)" % roc_auc_rf,"SVM (AUC=%0.2f)" % roc_auc_svm,"CART (AUC=%0.2f)" % roc_auc_cart], loc="lower right") 
plt.title("ROC Curves") 
plt.xlabel("False Positive Rate") 
plt.ylabel("True Positive Rate") 
plt.show()

# Testing the results on the unseen final test dataset.
# Algorithms performed:-
# 1. Logistic Regression
# 2. Decision Tree Classifier(CART)
# 3. Naive Bayes
# 4. Support Vector Machine(SVM)
# 5. Random Forest Classifier

# Though we have chosen Logistic Regression over other algorithms based on ROC curve(AUC = 0.81), we were curious to try our model and 
# check the accuracy with other algorithms too.

# Read the test dataset file.
df_ncaa_test = pd.read_csv("D:\Stevens_2nd Sem\Statistical Learning\NCAA - Project\Final_NCAA_Test.csv")
df_ncaa_test.head()

# Checking whether there are any Null values in the dataset. No Null values are there.
df_ncaa_test.isnull().sum()
# Dropping the variables
df_ncaa_test = df_ncaa_test.drop(['Index','round','teamA','TeamName_x','scoreA','teamB','TeamName_y','scoreB'],axis=1)
df_ncaa_test.head()

# Categorical values of 'winner' A and B are converted to discrete values of 0 and 1.
df_ncaa_test['winner_binary'] = df_ncaa_test['winner'].apply(lambda x: 0 if x == 'A' else 1)
del df_ncaa_test['winner']
df_ncaa_test.rename(columns={'winner_binary':'winner'}, inplace=True)
df_ncaa_test.head()
# Separating the test_features and test_target variables.
# test_features contain columns from 'fgm_diff' to 'de_oe_diff'. And test_target variable contain column 'winnner'. 
test_features = df_ncaa_test.ix[:,'fgm_diff':'de_oe_diff']
test_target = df_ncaa_test[['winner']]
test_features.shape, test_target.shape    # Dimensions of test_features and test_target set

# Plot Confusion Matrix
def plot_confusion(cm):
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title('Confusion matrix')
    plt.set_cmap('Blues')
    plt.colorbar()

    target_names = ['Winner', 'Loser']

    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=60)
    plt.yticks(tick_marks, target_names)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Convenience function to adjust plot parameters for a clear layout.
    plt.tight_layout()
    
# Logistic Regression
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")     # Ignore Warnings
preds_logreg_test = clf_log.predict(test_features)
print preds_logreg_test
probs = clf_log.predict_proba(test_features.astype(float))
probs
#print pd.crosstab(test_target, preds_cart, rownames=['actual'], 
#            colnames=['prediction'])
# summarize the fit of the model
print "Classification Report:", (metrics.classification_report(test_target,preds_logreg_test))
print "Confusion Matrix: ", (metrics.confusion_matrix(test_target,preds_logreg_test))
print "Accuracy Score for Logistic Regression is :", metrics.accuracy_score(test_target,preds_logreg_test)
print "AUC for Logistic Regression is:", roc_auc_score(test_target, preds_logreg_test)
print "Test Error Rate: ",zero_one_loss(test_target, preds_logreg_test)
print plot_roc_curve(test_target,probs[:,1])
print plot_prre_curve(test_target,probs[:,1])
plot_confusion(metrics.confusion_matrix(test_target,preds_logreg_test))

# Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

preds_cart = clf_cart.predict(test_features)
print preds_cart
probs = clf_cart.predict_proba(test_features.astype(float))
probs
print pd.crosstab(test_target, preds_cart, rownames=['actual'], 
            colnames=['prediction'])
# summarize the fit of the model
print "Classification Report:", (metrics.classification_report(test_target,preds_cart))
print "Confusion Matrix:", (metrics.confusion_matrix(test_target,preds_cart))
print "Accuracy Score for CART is :", metrics.accuracy_score(test_target,preds_cart)
print "AUC for CART is:", roc_auc_score(test_target, preds_cart)
print "Test Error Rate: ",zero_one_loss(test_target, preds_cart)
print plot_roc_curve(test_target,probs[:,1])
print plot_prre_curve(test_target,probs[:,1])
plot_confusion(metrics.confusion_matrix(test_target,preds_cart))

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score

preds_gnb = gnb.predict(test_features)
probs = gnb.predict_proba(test_features.astype(float))
probs
# summarize the fit of the model
print "Classification Report: ", (metrics.classification_report(test_target,preds_gnb))
print "Confusion Matrix:", (metrics.confusion_matrix(test_target,preds_gnb))
print "Accuracy Score for Naive Bayes is:", metrics.accuracy_score(test_target,preds_gnb)
print "AUC for Naive Bayes is:", roc_auc_score(test_target, preds_gnb)
print "Test Error Rate: ",zero_one_loss(test_target, preds_gnb)
print plot_roc_curve(test_target,probs[:,1])
print plot_prre_curve(test_target,probs[:,1])
plot_confusion(metrics.confusion_matrix(test_target,preds_gnb))

# Support Vector Machine
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import zero_one_loss
from sklearn.metrics import roc_auc_score

# make predictions
preds_svm = clf_svm.predict(test_features)
probs = clf_svm.predict_proba(test_features.astype(float))
probs
print preds_svm
# summarize the fit of the model
print "Classification Report: ", (metrics.classification_report(test_target,preds_svm))
print "Confusion Matrix: ", (metrics.confusion_matrix(test_target,preds_svm))
print "Accuracy Score:", metrics.accuracy_score(test_target,preds_svm)
print "AUC for SVM is:", roc_auc_score(test_target, preds_svm)
print "Test Error Rate: ",zero_one_loss(test_target, preds_svm)
print plot_roc_curve(test_target,probs[:,1])
print plot_prre_curve(test_target,probs[:,1])
plot_confusion(metrics.confusion_matrix(test_target,preds_svm))

# RandomForest Classification

from sklearn.ensemble import RandomForestClassifier

preds_rf = clf_rf.predict(test_features)
print preds_rf
probs = clf_rf.predict_proba(test_features.astype(float))
probs
print "Classification Report: ", (metrics.classification_report(test_target,preds_rf))
print "Confusion Matrix: ", (metrics.confusion_matrix(test_target,preds_rf))
print "Accuracy Score:", metrics.accuracy_score(test_target,preds_rf)
print "AUC for Random Forest is:", roc_auc_score(test_target, preds_rf)
print "Test Error Rate: ",zero_one_loss(test_target, preds_rf)
print plot_roc_curve(test_target,probs[:,1])
print plot_prre_curve(test_target,probs[:,1])
#-----------------------------------------------------------------------------------------
print "Classification Score on Validation set:", clf_rf.score(test_features,test_target)
#-----------------------------------------------------------------------------------------
plot_confusion(metrics.confusion_matrix(test_target,preds_rf))