import os
import sys
from preprocessing import PreProcessor
from training import NaiveBayesClassifier
from analysis import Optimize
import matplotlib.pyplot as plt


stopwords_path = 'stopwords.txt'
special_characters = set(['!', '\"', '#', '$', '%', '&', '\'', '(', ')', '*', '+', ',', '-', '.', '/',
                       ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~'])

"""------------Task 1: Feature Selection------------------"""
print("======================================Task 1: Feature Selection======================================")

# make a preprocess object to preprocess the taining data using stapword file and special characters as input
myPerProcess = PreProcessor(stopwords_path, special_characters)            # make a preprocess object
features = myPerProcess.process_file('train.csv')                          # save a 1000 most recently appeared words as features

# print the 20 most recently appeared words
myPerProcess.top_20_words(features)                            

input("Press Enter to continue...")

"""------------Task 2: Model Training and Evaluation------------------"""
print("\n\n======================================Task 2: Model Training and Evaluation======================================")

# NaiveBayesClassifier class contains a training code and prediction code
# input: feature(1000 words), PreProcessor object
myClassifier = NaiveBayesClassifier(features, myPerProcess)             # make a NaiveBayesClassifier object 

# for task 2 I randomly choose the data set size to be 100% and laplace smoothing parameter to be 3
# at task 3 I will optimize using various value
data_ratio = 1
k = 2000

# train the model using train() method
myClassifier.train(data_ratio, k)

# using the model trained evaluate the performance using predict() method
accuracy = myClassifier.predict('test.csv')


input("Press Enter to continue...")

"""------------Task 3: Learning Curve Analysis------------------"""
print("\n\n======================================Task 3: Learning Curve Analysis======================================")

# Optimize class contains a method to optimize the model and other funtions to visialize and evaluate the performance
optimize = Optimize(features, myPerProcess)

# at plot_learning_curve funtion I try various data set size and returns the size with best accuracy
best_training_size = optimize.plot_learning_curve()     

# at plot_laplace_curve funtion I try various laplace parameter and plot the performance
optimize.plot_laplace_curve(best_training_size)

input("Press Enter to exit...")
