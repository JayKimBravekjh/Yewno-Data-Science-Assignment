# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:11:45 2016

@author: jaykim
"""

import pandas as pd
from pandas import Series, DataFrame

import numpy as np
import matplotlib.pyplot as plt

# data import 
train_df = pd.read_csv("train.csv", dtype={"Age": np.float64}, )
test_df = pd.read_csv("test.csv", dtype={"Age":np.float64}, )

# data display (first 5 lines)
print(train_df.head())
print(test_df.head())

# acquires labels from Embarked in test_df
test_df = test_df.join(pd.get_dummies(test_df.Embarked, prefix = 'Emb'))
# acuires labels from Embarked.in train_df 
train_df = train_df.join(pd.get_dummies(test_df.Embarked, prefix='Emb'))

# acquires labels from Sex in test_df
test_df = test_df.join(pd.get_dummies(test_df.Sex, prefix = 'Sex'))
# acuqires labels from Sex in train_df
train_df = train_df.join(pd.get_dummies(train_df.Sex, prefix = 'Sex'))

# dropping some features 
test_df = test_df.drop(['Embarked', 'Sex', 'Name', 'Ticket', 'Cabin'], axis = 1)
train_df = train_df.drop(['Embarked', 'Sex', 'Name', 'Ticket', 'Cabin'], axis = 1)
print(train_df.describe())

# median of age. 
median_age = train_df.Age.median(axis=0)
# filling missing data or non-number value with median age. in  train data. 
train_df.Age = train_df.Age.fillna(median_age)
# filling  missing data or non-number value with median age. in test data. 
test_df.Age = test_df.Age.fillna(median_age)

# median fare 
median_fare = train_df.Fare.median(axis=0)
# filling missing data or non-number value with median fare. in train data. 
train_df.Fare = train_df.Fare.fillna(median_fare)
# filling missing data or non-number value with median fare in test data. 
test_df.Fare = test_df.Fare.fillna(median_fare)

# index values 
y_train_orig = train_df.iloc[:,1].values
# rest train values
X_train_orig = train_df.iloc[: ,[2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
# rest test values 
X_test_orig = test_df.iloc[:,[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]

# train data column names 
train_col = X_train_orig.columns
# test data column names 
test_col = X_test_orig.columns

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
# put into the data frame. for X-test data 
X_test_orig = pd.DataFrame(sc.fit_transform(X_test_orig))
# interpolate the missing values using two values sides. 
X_train_orig = X_train_orig.interpolate()
# put into the data frame for X-train data. 
X_train_orig = pd.DataFrame(sc.fit_transform(X_train_orig))
# test column names
X_test_orig.columns = test_col
# train column names 
X_train_orig.columns = train_col



# Importance rate calculation using RandomForestClassifier. 
from sklearn.ensemble import RandomForestClassifier 
feat_labels = X_train_orig.columns 
forest = RandomForestClassifier(n_estimators=1000, random_state=0, n_jobs=-1)
forest.fit(X_train_orig, y_train_orig)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print(importances)
print(indices)

for f in range(X_train_orig.shape[1]):
    print("%2d) %-*s %f" % (f + 1, 30, feat_labels[indices[f]], importances[indices[f]]))
    
X_train_orig = forest.transform(X_train_orig, threshold=.05)
X_test_orig = forest.transform(X_test_orig, threshold=.05)
X_train_orig 

## learning curve of GaussianNB
import matplotlib.pyplot as plt 
from sklearn.learning_curve import learning_curve
from sklearn.naive_bayes import GaussianNB
print("GaussianNB")
train_sizes, train_scores, test_scores = learning_curve(estimator=GaussianNB(), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_orig, y_train_orig, test_size=0.25, random_state=0)


gaussian = GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
print(gaussian.score(X_test, y_test))

## learning curves of LogisticRegression 
from sklearn.linear_model import LogisticRegression
print("LogisticRegression")
train_sizes, train_scores, test_scores = learning_curve(estimator=LogisticRegression(), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()


logreg = LogisticRegression()
logreg.fit(X_train, y_train)
Y_pred = logreg.predict(X_test)
print(logreg.score(X_test, y_test))

## learning curves of KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier

print("KNeighborsClassifier")
train_sizes, train_scores, test_scores = learning_curve(estimator=KNeighborsClassifier(n_neighbors=3), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()


#coeff_df = DataFrame(train_df.columns.delete(0))
#coeff_df.columns = ['Features']
#coeff_df["Coefficient Estimate"] = pd.Series(logreg.coef_[0])
#print(coeff_df)

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, y_train)
Y_pred = knn.predict(X_test)
print(knn.score(X_test, y_test))

#from sklearn.metrics import classification_report
#from sklearn import metrics
#y_true, y_pred = y_test, clf.predict(X_test)
#print(classification_report(y_true, y_pred))
#y_pred = clf.predict(X_test).astype(int)

## learning curve of SVM 
from sklearn.svm import SVC
svm = SVC(kernel='rbf', C= 100.0, gamma = 0.1, random_state=0 )

print("SVM")
train_sizes, train_scores, test_scores = learning_curve(estimator=svm, X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()



svm = SVC(kernel= 'rbf', C = 100.0, gamma = 0.1, random_state=0)
svm.fit(X_train, y_train)
Y_pred = svm.predict(X_test)
print(svm.score(X_test, y_test))



## learning curve of validation curve for parameter. 
from sklearn.learning_curve import validation_curve
print("parameter check for SVM")
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve(estimator=svm, X=X_train_orig, y=y_train_orig, param_name='C', param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label = 'training accumracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha = 0.15, color='blue')
plt.plot(param_range, test_mean, color = 'green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')
plt.xscale('log')
plt.grid()
plt.xlabel('Parameter')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.85])
plt.show()


## learning curve of RandomForestClassifier 
print("RandomForestClassifier")
train_sizes, train_scores, test_scores = learning_curve(estimator=RandomForestClassifier(), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()



from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators = 200, 
    min_samples_split = 4, 
    min_samples_leaf=2)
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))



from sklearn.ensemble import ExtraTreesClassifier
#from sklearn.tree import ExtraTreeClassifier
# learning curve of ExtraTreesClassifier 
print("ExtraTreesClassifier")
train_sizes, train_scores, test_scores = learning_curve(estimator=ExtraTreesClassifier(), X=X_train_orig, y=y_train_orig, cv=10, n_jobs=1)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)
plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5, label = 'training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha = 0.15, color = 'blue')
plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha = 0.15, color= 'green')
plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.7, 0.9])
plt.show()


from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators = 300, max_depth = None, min_samples_split=1, random_state=0)
clf.fit(X_train, y_train)
Y_pred = clf.predict(X_test)
print(clf.score(X_test, y_test))


print("grid search CV")

## Grid Search Cross Validation. 
from sklearn.grid_search import GridSearchCV
param_grid = [{'C' : param_range, 'kernel':['linear']}, {'C':param_range, 'gamma':param_range, 'kernel':['rbf']}]
gs = GridSearchCV(estimator=svm, param_grid=param_grid, scoring='accuracy', cv=10, n_jobs=1)
gs = gs.fit(X_train_orig, y_train_orig)
print(gs.best_score_)

print(gs.best_params_)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train_orig, y_train_orig, test_size=0.25, random_state=0)
## svm gamma tests. 

## gamma = 0.1 is chosen. 
svm = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
no_samples = len(y_test)
print('Misclassified samples: %d of %d' % ((y_test != y_pred).sum(), no_samples))


svm = SVC(kernel='rbf', C=100.0, gamma=1, random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
no_samples = len(y_test)
print('Misclassified samples: %d of %d' % ((y_test != y_pred).sum(), no_samples))



svm = SVC(kernel='rbf', C=100.0, gamma=10, random_state=0)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_test)
no_samples = len(y_test)
print('Misclassified samples: %d of %d' % ((y_test != y_pred).sum(), no_samples))



from sklearn.metrics import accuracy_score
print ('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

## confusion matrix .
from sklearn.metrics import confusion_matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
## precision, recall, F-1 tests. 
from sklearn.metrics import classification_report
from sklearn import metrics
y_true, y_pred = y_test, svm.predict(X_test)
print(classification_report(y_true, y_pred))

#y_pred = clf.predict(X_test).astype(int)

## graph of confusion metrics 
fig, ax = plt.subplots(figsize = (2.5, 2.5))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[i]):
        ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

## 10 times repeating cross validation 
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(estimator=svm, X=X_train, y=y_train, cv=10, n_jobs=1)
print('CV accuracy scores: %s' %scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


svmorig = SVC(kernel='rbf', C=100.0, gamma=0.1, random_state=0)
svmorig.fit(X_train_orig, y_train_orig)

y_pred_orig = svm.predict(X_test_orig)
y_pred_orig.sum()

output = test_df.PassengerId
output = pd.DataFrame(output)

predict = pd.DataFrame(y_pred_orig)
output = output.join(predict)
output.columns = ['PassengerId', 'Survived']
print(output)


import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Age versus Survival rate. graph. 
sns.factorplot('Age', 'Survived', data = train_df, size=4, aspect=3)

fig, (axis1, axis2, axis3) = plt.subplots(1, 3, figsize = (15, 5))
sns.countplot(x='Age', data = train_df, ax=axis1)
sns.countplot(x='Survived', hue="Age", data = train_df, order=[1, 0], ax=axis2)

Age_perc = train_df[["Age", "Survived"]].groupby(['Age'], as_index=False).mean()
sns.barplot(x='Age', y='Survived', data=Age_perc, ax=axis3)

Age_dummies_titanic = pd.get_dummies(train_df['Age'])
Age_dummies_test = pd.get_dummies(test_df['Age'])


train_df = train_df.join(Age_dummies_titanic)
test_df = test_df.join(Age_dummies_test)

#fig, (axis1, axis2) = plt.subplots(1, 2, figsize=(15, 4))
#axis1.set_title('Original Age values - Titanic')
#axis2.set_title('New Age values - Titanic')

average_age_titanic = train_df["Age"].mean()
std_age_titanic = train_df["Age"].std()
count_nan_age_titanic = train_df["Age"].isnull().sum()

average_age_test = test_df["Age"].mean()
std_age_test = test_df["Age"].std()
count_nan_age_test = test_df["Age"].isnull().sum()


facet = sns.FacetGrid(train_df, hue="Survived", aspect=4)
facet.map(sns.kdeplot, 'Age', shade = True)
facet.set(xlim=(0, train_df['Age'].max()))
facet.add_legend()

#fig, axis1 = plt.subplots(1, 1, figsize = (18, 4))
#average_age = train_df[["Age", "Survived"]].grouby(['Age'], as_index=False).mean()
#sns.barplot(x='Age', y='Survived', data = average_age)

#def chile_female_male(passenger):
#    Age, Sex = passenger
#    if Age < 16:
#        return 'child'
#    else:
#        return Sex

#train_df['Type'] = train_df[['Age']].apply(child_female_male, axis=1)
#test_df['Type'] = test_df[['Age']].apply(child_female_male, axis=1)

#train_df["Emb_C"] = train_df["Emb_C"].fillna("S")

#sns.factorplot('Emb_C', data=train_df, kind='count')
#sns.factorplot('Type', data = train_df, kind="count", palette='summer')
#sns.factorplot('Pclass', data = train_df, kind='count', hue='Type', x_order=(1, 2, 3), palette = 'winter')
#### graph of Age, Pclass versus survival rate. 
fig = sns.FacetGrid(train_df, hue='Pclass', aspect = 4)
fig.map(sns.kdeplot, 'Age', shade = True)
fig.set(xlim=(0, train_df['Age'].max()))
fig.add_legend()


## fare versus survival rate. 
fare_not_survived = train_df["Fare"][train_df["Survived"] == 0]
fare_survived = train_df["Fare"][train_df["Survived"] == 1] 

average_fare = DataFrame([fare_not_survived.mean(), fare_survived.mean()])
std_fare = DataFrame([fare_not_survived.std(), fare_survived.std()])

train_df['Fare'].plot(kind='hist', figsize=(15, 3), bins=100, xlim= (0, 50))

average_fare.index.names = std_fare.index.names = ["Survived"]
average_fare.plot(yerr=std_fare, kind='bar', legend=False)


## family size calculation.
train_df['Family'] = train_df["Parch"] + train_df["SibSp"]
train_df['Family'].loc[train_df['Family'] > 0 ] = 1
train_df['Family'].loc[train_df['Family'] == 0] = 0

#test_df['Family'] = test_df.drop(['SibSp', 'Parch'], axis=1)
#test_df = test_df.drop(['SibSp', 'Parch'], axis=1)

## graph of family size versus survival rate. 
fig, (axis1, axis2) = plt.subplots(1, 2, sharex=True, figsize = (10, 5))
sns.countplot(x='Family', data=train_df, order=[1,0],  ax=axis1)

family_perc = train_df[["Family", "Survived"]].groupby(['Family'], as_index=False).mean()
sns.barplot(x='Family', y='Survived', data=family_perc, order=[1, 0], ax=axis2)

axis1.set_xticklabels(["With Family", "Alone"], rotation=0)

sns.lmplot('Age', 'Survived', hue='Family', data = train_df)
#def get_person(passenger):
#    age, sex = passenger
#    return 'child' if age < 18 else sex
    
#train_df['Person'] = train_df[['Age']].apply(get_person, axis=1)
#test_df['Person'] = test_df[['Age']].apply(get_person, axis=1)

#train_df.drop(['Sex'], axis=1, inplace = True)
#test_df.drop(['Sex'], axis=1, inplace=True)

print(__doc__)
