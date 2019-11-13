# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 02:54:37 2016

@author: Akindolu Dada
"""

"Titanic Kaggle Competition"

# Import the Pandas library
import pandas as pd

# Import the Numpy library
import numpy as np

# Import 'tree' from scikit-learn library
from sklearn import tree

# Import the `RandomForestClassifier`
from sklearn.ensemble import RandomForestClassifier

# Import random  library
import random

# Load the train and test datasets to create two DataFrames
train_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/train.csv"
train = pd.read_csv(train_url)

test_url = "http://s3.amazonaws.com/assets.datacamp.com/course/Kaggle/test.csv"
test = pd.read_csv(test_url)

#Print the `head` of the train and test dataframes
#print(train.head())
#print(test.head())

#train = train.dropna(subset=["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"])

# Convert the male and female groups to integer form
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

# Impute the median age for na values
train["Age"] = train["Age"].fillna(train["Age"].median())
test["Age"] = test["Age"].fillna(test["Age"].median())

# Impute the Embarked variable
train["Embarked"] = train["Embarked"].fillna("S")
test["Embarked"] = test["Embarked"].fillna("S")

# Convert the Embarked classes to integer form
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2

# Replace nan by median
test["Fare"][test["Fare"].isnull()] = test["Fare"].median()

#Print the Sex and Embarked columns
#print(train["Sex"])
#print(train["Embarked"])

nRows = train.shape[0]
nColumns = train.shape[1]
validateRatio = 0.2
nValidate = int(validateRatio*nRows)
nTrain = nRows - nValidate

validateIndex = random.sample(range(nRows), nValidate)
trainIndex = np.array(list(set(np.arange(0,nRows)) - set(validateIndex)))
validateData = train.iloc[validateIndex].copy()
trainData = train.iloc[trainIndex].copy()

# Create the target and features numpy arrays: target, features_one
treeFeatureNames = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
treeTrainTarget = trainData["Survived"].values
treeTrainFeatures = trainData[treeFeatureNames].values
treeValidTarget = validateData["Survived"].values
treeValidFeatures = validateData[treeFeatureNames].values

# Fit your first decision tree: my_tree_one
my_tree_one = tree.DecisionTreeClassifier(max_depth = 4, min_samples_split = 10, random_state = 1)
my_tree_one = my_tree_one.fit(treeTrainFeatures, treeTrainTarget)
treeValidPrediction = my_tree_one.predict(treeValidFeatures)
treeValidPredAcc = sum(treeValidTarget == treeValidPrediction)/nValidate

# Look at the importance and score of the included features
#print(my_tree_one.feature_importances_)
#print(my_tree_one.score(treeTrainFeatures, treeTrainTarget))
print(treeValidPredAcc)

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
treeTestFeatures = test[treeFeatureNames].values

# Make your prediction using the test set
treeTestPrediction = my_tree_one.predict(treeTestFeatures)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
treeSolution = pd.DataFrame(treeTestPrediction, PassengerId, columns = ["Survived"])
#print(treeSolution)

# Check that your data frame has 418 entries
#print(treeSolution.shape)

# Write your solution to a csv file with the name my_solution.csv
treeSolution.to_csv("treeSolution_one.csv", index_label = ["PassengerId"])






# We want the Pclass, Age, Sex, Fare,SibSp, Parch, and Embarked variables
forestTrainFeatures = trainData[treeFeatureNames].values
forestTrainTarget = trainData["Survived"].values
forestValidTarget = validateData["Survived"].values
forestValidFeatures = validateData[treeFeatureNames].values

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 8, min_samples_split=10, n_estimators = 100, random_state = 0)
my_forest = forest.fit(forestTrainFeatures, forestTrainTarget)
forestValidPrediction = my_forest.predict(forestValidFeatures)
forestValidPredAcc = sum(forestValidTarget == forestValidPrediction)/nValidate

# Print the score of the fitted random forest
#print(my_forest.score(forestTrainFeatures, forestTrainTarget))
print(forestValidPredAcc)

# Compute predictions on our test set features then print the length of the prediction vector
forestTestFeatures = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forestTestPrediction = my_forest.predict(forestTestFeatures)
#print(len(forestTestPrediction))

forestSolution = pd.DataFrame(forestTestPrediction, PassengerId, columns = ["Survived"])
#print(treeSolution)

# Check that your data frame has 418 entries
#print(forestSolution.shape)

# Write your solution to a csv file with the name my_solution.csv
forestSolution.to_csv("forestSolution_one.csv", index_label = ["PassengerId"])
