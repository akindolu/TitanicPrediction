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

# Import random  library
import random

# Import 'MLPClassifier' from scikit-learn library
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler  

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


nRows = train.shape[0]
cvRatio = 0.2
nCV = int(cvRatio*nRows)
nTrain = nRows - nCV
cvIndex = random.sample(range(nRows), nCV)
trainIndex = np.array(list(set(np.arange(0,nRows)) - set(cvIndex)))
trainData = train.iloc[trainIndex].copy()
cvData = train.iloc[cvIndex].copy()

# Create the target and features numpy arrays: target, features_one
#featureNames = ["Pclass", "Sex", "Age", "Fare", "SibSp", "Parch", "Embarked"]
featureNames = ["Pclass", "Sex", "Age", "Fare"]
trainTarget = trainData["Survived"].values
trainFeatures = trainData[featureNames].values
cvTarget = cvData["Survived"].values
cvFeatures = cvData[featureNames].values


# Scale the input features
scaler = StandardScaler()
scaler.fit(trainFeatures)
scaledTrainFea = scaler.transform(trainFeatures)
scaledCVFea = scaler.transform(cvFeatures)

# Fit your first decision tree: my_tree_one
nnModel = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(5, 20, 20, 2), random_state=1)
nnModel.fit(scaledTrainFea, trainTarget)
print(nnModel.score(scaledCVFea, cvTarget))

# Extract the features from the test set: Pclass, Sex, Age, and Fare.
testFeatures = test[featureNames].values

# Scale the test features
scaledTestFea = scaler.transform(testFeatures)

# Make your prediction using the test set
nnPrediction = nnModel.predict(scaledTestFea)


# Create a data frame with two columns: PassengerId & Survived. Survived contains your predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
nnSolution = pd.DataFrame(nnPrediction, PassengerId, columns = ["Survived"])
#print(treeSolution)

# Check that your data frame has 418 entries
#print(treeSolution.shape)

# Write your solution to a csv file with the name my_solution.csv
nnSolution.to_csv("nnSolution_one.csv", index_label = ["PassengerId"])






