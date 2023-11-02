import csv
from collections import Counter
#-------------------Task 2: Model Training and Evaluation----------------------

"""Using "train.csv" and "test.csv", which they will use to train and evaluate
their Naive Bayes Classifier with Laplace Smoothing

*Laplace Smoothing: Implement Laplace smoothing in the parameter estimation. 
For an attribute Xi with k values, Laplace correction adds 1 to the numerator and k to the denominator of the maximum likelihood estimate.

- Evaluation measure: Accuracy
- Please describe your observations and provide an analysis of their model's performance.
"""

class NaiveBayesClassifier:
    def __init__(self, features, myPerProcess):
        # Initialize the class with a list of features and an instance of the pre-processing class
        self.features = features
        self.word_probs = []  # This will store the probabilities of each word in each class
        self.train_file_path = 'train.csv'  # File path to the training data
        self.myPerProcess = myPerProcess  # Instance of the pre-processing class

    def train(self, train_data_ratio, k):
        print("\n<<<Training>>>")
        print("Loading stopwords...")
        self.myPerProcess.load_stopwords()  # Load the stop words for pre-processing

        # in this part the we count the 5-star and 1-star review in all the data
        print("Counting positive and negative instances in training set...")
        # variable to count the 5 star, 1 star rating in every features
        pos_count = 0
        neg_count = 0
        with open(self.train_file_path) as f:
            reader = csv.DictReader(f)

            # Read in the rows from the training data
            rows = [row for row in reader]
            num_rows = len(rows)
            train_rows = rows[:int(train_data_ratio*num_rows)]
            for row in train_rows:
                # Count the positive instances (5-star reviews)
                if row['stars'] == '5': 
                    pos_count += 1
                # Count the negative instances (1 star reviews)
                else:
                    neg_count += 1

        # in this part the we count the occurrences of each word in 5-star and 1-star review
        print("Counting occurrences of each feature in positive and negative instances...")
        pos_word_count = Counter()
        neg_word_count = Counter()
        for row in train_rows:
            # Count the occurrences of each word in positive instances
            if row['stars'] == '5':
                pos_word_count.update(self.myPerProcess.preprocess_text(row['text']))
            # Count the occurrences of each word in negative instances
            else:
                neg_word_count.update(self.myPerProcess.preprocess_text(row['text']))

        # in this part the calculate 5-star and 1-star probability in every feature
        print("Computing probabilities of each feature given the target class...")
        for word in self.features:
            # Compute the probability of each word given the target class using Laplace smoothing
            pos_word_prob = (pos_word_count[word] + 1) / (pos_count + k)
            neg_word_prob = (neg_word_count[word] + 1) / (neg_count + k)
            
            # Add the word probabilities to the list
            self.word_probs.append((word, pos_word_prob, neg_word_prob)) 


    def predict(self, test_file_path):
        print("\n<<<Predicting>>>")
        # Load stopwords
        self.myPerProcess.load_stopwords()

        # Preprocess the test data and store it in a list
        preprocessed_texts = []
        with open(test_file_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                preprocessed_texts.append(self.myPerProcess.preprocess_text(row['text']))

        # Initialize true positive, false positive, true negative, and false negative counters
        true_pos = 0
        false_pos = 0
        true_neg = 0
        false_neg = 0
        
        # Open the test file and iterate over each row
        with open(test_file_path) as f:
            reader = csv.DictReader(f)
            counter = 0
            for i, row in enumerate(reader):
                # Initialize the probability ratios for positive and negative classes
                pos_ratio = 1
                neg_ratio = 1
                # Iterate over each word in the preprocessed text and update the probability ratios
                for word, pos_word_prob, neg_word_prob in self.word_probs:
                    if word in preprocessed_texts[i]:
                        pos_ratio *= pos_word_prob
                        neg_ratio *= neg_word_prob
                    else:
                        pos_ratio *= 1 - pos_word_prob
                        neg_ratio *= 1 - neg_word_prob
                # Compute the probability of the review being positive and negative using the computed probability ratios
                # To avoid division by zero error, a very small value (1e-10) is added to the denominator
                pos_prob = pos_ratio / (pos_ratio + neg_ratio + 1e-10)
                neg_prob = neg_ratio / (pos_ratio + neg_ratio + 1e-10)
                # Make the final prediction based on the probability
                if pos_prob > neg_prob:
                    prediction = '5'
                else:
                    prediction = '1'
                # Update the true positive, false positive, true negative, and false negative counters based on the prediction
                if prediction == row['stars']:
                    if prediction == '5':
                        true_pos += 1
                    else:
                        true_neg += 1
                else:
                    if prediction == '5':
                        false_pos += 1
                    else:
                        false_neg += 1
                
                # Print the progress of prediction for every 250th review
                counter += 1
                if counter % 250 == 0:
                    print(f"Processed {int(counter/10)}% of test data...")

        # Compute accuracy, precision, recall, and f1 score based on 
        # the true positive, false positive, true negative, and false negative counters
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)
        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        f1 = 2 * precision * recall / (precision + recall)


        print("\n<<<Prediction Results>>>")
        print("==================================================")
        print(f"|              |   Actual Positive |   Actual Negative |")
        print(f"|--------------|------------------|------------------|")
        print(f"|Predicted Pos. |        {true_pos:3d}         |        {false_pos:3d}         |")
        print(f"|Predicted Neg. |        {false_neg:3d}         |        {true_neg:3d}         |")
        print("--------------------------------------------------")
        print(f"Accuracy:  {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall:    {recall:.2f}")
        print(f"F1 Score:  {f1:.2f}")

        return accuracy



