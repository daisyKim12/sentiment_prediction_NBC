# Naive Bayes Classifier for Sentiment Analysis
This project involves the development of a Naive Bayes Classifier (NBC) for sentiment analysis using a dataset that includes reviews along with their corresponding star ratings. The dataset comprises two files, "train.csv" and "test.csv," which will be supplied for this project. In this context, a review with a 5-star rating is categorized as expressing a positive sentiment, whereas all other ratings are treated as indicating a negative sentiment.

## Description

**Feature Selection:**   
- Preprocess "train.csv" dataset.
- Select the top 1000 most frequently occurring words as features for the Naive Bayes Classifier model.
- Exclude all other words not in the top 1000 by frequency.
- Print a list of the selected features, typically the top 20 to 50 words.

**Model Training and Evaluation:**   
- Utilize both "train.csv" and "test.csv" datasets.
- Train and evaluate the Naive Bayes Classifier.
- Apply Laplace Smoothing during parameter estimation.
- For an attribute Xi with k values, incorporate Laplace correction by adding 1 to the numerator and k to the denominator of the maximum likelihood estimate.

**Learning Curve Analysis**   
- Create a learning curve by varying the percentage of training data used (e.g., 10%, 30%, 50%, 70%, 100%).
- Keep the testing set unchanged.

## file discription
- main.py: main python file so in this file task is running. There are total 3 task in this project. And in main.py the task can be checked. In order to check the result just enter the following code in the terminal 
- preprocessing.py: this file contains a class having the variable and method to perform task1. To preprocess the data set
- training.py: this file contains a class having the variable and method to perform task2. This file contains Naive Bayes Classification training method and prediction method
- analysis.py: this file contains a class having the variable and method to perform task3. This file contains visualization and optimization methods.

## Result

**Tokenlixze**   
Tokenize all the review and get 1000 most frequently used words in the train data to use these words as a feature. And print 20 most frequently from 1000 words.   
![](x/Screenshot%202023-11-02%20at%2011.12.57%20PM.png)

**Train and predict new dataset**   
I trained the model using train data and using test data predict the rating (5- star, 1-star) and compare the result with real rating.   
![](x/Screenshot%202023-11-02%20at%2011.14.26%20PM.png)

**Visualize Learning Curve**   
![](x/Screenshot%202023-11-02%20at%2011.15.31%20PM.png)