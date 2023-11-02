import os
import sys
from training import NaiveBayesClassifier

import matplotlib.pyplot as plt

#-------------------Task 3: Learning Curve Analysis----------------------

"""Students will plot a learning curve by varying the amount of training data used [10%, 30%, 50%, 70%, 
100%]. The testing set will remain unchanged.

*For this plotting task only, students may use external plotting packages like the Matplotlib.
*Students will describe their observations and provide an analysis of the learning curve."""

class Optimize:
    def __init__(self, features, myPerProcess):
        self.features = features
        self.myPerProcess = myPerProcess
        
    
    def plot_learning_curve(self):
        training_sizes = [0.1, 0.3, 0.5, 0.7, 1] # fraction of total dataset
        accuracy_scores = []
        k = 1024

        for ratio in training_sizes:
            print(f"\n<<<{int(ratio * 100)}% data>>>")

            classifier = NaiveBayesClassifier(self.features, self.myPerProcess)

            # Train the classifier on the specified number of training samples
            classifier.train(ratio, k)

            # Test the classifier on the test dataset and record the accuracy score
            score = classifier.predict('test.csv')
            accuracy_scores.append(score)

        # Find the index of the maximum accuracy score and use it to get the corresponding training size
        max_score = max(accuracy_scores)
        max_index = accuracy_scores.index(max_score)
        best_training_size = training_sizes[max_index]

        # Plot the learning curve
        plt.plot(training_sizes, accuracy_scores, '-o')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curve Analysis')
        plt.show()

        return best_training_size
    

    def plot_laplace_curve(self, best_training_size):
    
        laplace_param = [1, 4, 16, 64, 256, 1024, 4096]
        accuracy_scores = []

        for k in laplace_param:
            print(f"\n<<<k = {k}>>>")
            classifier = NaiveBayesClassifier(self.features, self.myPerProcess)

            # Train the classifier on the specified laplace_param
            classifier.train(best_training_size, k)

            # Test the classifier on the test dataset and record the accuracy score
            score = classifier.predict('test.csv')
            accuracy_scores.append(score)

        # Plot the laplace curve
        plt.plot(laplace_param, accuracy_scores, '-o')
        plt.xlabel('Laplace Parameter')
        plt.ylabel('Accuracy Score')
        plt.title('Laplace Curve Analysis')
        plt.show()

    







