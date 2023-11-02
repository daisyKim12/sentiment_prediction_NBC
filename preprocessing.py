import csv
from collections import Counter

#-------------------Task 1: Feature Selection----------------------

"""Students will preprocess "train.csv" and select the top 1000 words (by frequency) as word features 
for their model. All other words will be ignored.
Please print out the top 20-50 words from the selected features.

* Preprocessing Guideline:
a. Convert all text to lowercase
b. Remove special characters.
c. Tokenize the text into words.
d. Remove stop words.
e. Select 1000 most frequently appeared words for the final features"""


class PreProcessor:
    def __init__(self, stopwords_path, special_characters):
        self.stopwords_path = stopwords_path                        # stopwords file path
        self.special_characters = special_characters                # set of special characters to be removed
        self.stop_words = set()                                     # an empty set to hold the stop words
        
    def load_stopwords(self):
        with open(self.stopwords_path, 'r') as f:                   # open the stopwords file in read mode
            self.stop_words.update([word.strip() for word in f])    # add the stop words to the set

    def preprocess_text(self, text):
        # remove special characters
        clean_text = ''
        for char in text:
            if char not in self.special_characters:
                clean_text += char

        # split into words
        words = clean_text.split()

        # remove stop words
        words = [w for w in words if w not in self.stop_words]
        return words
    
    def process_file(self, file_path):
        self.load_stopwords()                                       # load the stop words
        
        # preprocess the text and count word frequencies
        word_counts = Counter()                                     # create a counter to count word frequencies
        
        # open the input file
        with open(file_path) as f:                                  
            reader = csv.DictReader(f)                              # create a dictionary reader object
            
            # iterate over each row in the file
            for row in reader:                                     
                text = row['text'].lower()                          # get the text in the current row and convert to lowercase
                words = self.preprocess_text(text)                  # preprocess the text in the current row
                word_counts.update(words)                           # update the counter with the preprocessed words

        # select the top 1000 words by frequency as word features
        word_features = [w for w, _ in word_counts.most_common(1000)]       # get the 1000 most common words from the counter
        return word_features                                                # return the selected word features

    def top_20_words(self, word_features):
        print("\n<<<Top 20 words>>>")
        print(word_features[:20])                   # print the top 20 words
        


