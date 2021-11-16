# JSON Reader import
import json
from os import error
from re import VERBOSE
from scipy.sparse import data
# SKLearn Vectorizer
from sklearn.feature_extraction.text import CountVectorizer
# Numpy
import numpy as np
# SkLearn Classifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
# Sklearn Train Test Split
from sklearn.model_selection import train_test_split
# Sklearn plotconfusionmatrix
from sklearn.metrics import plot_confusion_matrix
# Matplotlib
import matplotlib.pyplot as plt
# Pickle
import pickle
# Pandas
import pandas as pd

# Validation
validation_csv_location = 'Competition_Files/validation_data.csv'
validation_rows = 500

# Parameters
file_location = "Sentiment_140/sentiment_train_cleaned.csv"
max_entries = 1599996
# Models
model_vectorizer_number = 1

# Paths for Model and Vectorizer
model_name = f'Model_{model_vectorizer_number}/model_{model_vectorizer_number}.pkl'
vectorizer_name = f'Model_{model_vectorizer_number}/vectorizer_{model_vectorizer_number}.pkl'

# Main Reader
def vectorize_data(data):
    # Data Length
    print(f'Data Length: {len(data)}')
    # Instance of CountVectorizer()
    vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2))
    # Vectorize Data
    vectorized_data = vectorizer.fit_transform(data)
    # Save Vectorizer
    save_vectorizer(vectorizer)
    # Shape
    print(f'Vectorized Data Shape: {vectorized_data.shape}')

    # Return Vectorized Data
    return vectorized_data

def train_classifier(data_x, data_y):
    # Instance of MultinomialNB()
    classifier = MultinomialNB()
    # Logistic Regression Test
    # classifier = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=20000)

    # Train Test Split
    train_x, test_x, train_y, test_y = train_test_split(
        data_x, data_y, test_size=0.2, shuffle=True, random_state=42)
    # Fit
    classifier.fit(train_x, train_y)
    # Test Accuracy
    accuracy = test_accuracy(classifier, test_x, test_y)
    print(accuracy)
    # Return Classifier
    return classifier

def test_accuracy(classifier, data_x, data_y):
    # Predict
    predictions = classifier.predict(data_x)
    # Accuracy
    accuracy = np.mean(predictions == data_y)
    # Plot Confusion Matrix
    plot_confusion_matrix(classifier, data_x, data_y)
    plt.show()
    # Return Accuracy
    return accuracy

def test_individual_sentiments(model, vectorizer, data):
    # Apply Transformations
    vectorized_data = vectorizer.transform(data)
    # Predict
    predictions = model.predict_proba(vectorized_data)
    # Return Predictions (0 = Negative, 1 = Positive)
    return predictions[0][1]

def save_model(model):
    # Save Model
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)

def save_vectorizer(vectorizer):
    # Save Vectorizer
    with open(vectorizer_name, 'wb') as f:
        pickle.dump(vectorizer, f)

def load_model(model_name=model_name):
    # Load Model
    with open(model_name, 'rb') as f:
        model = pickle.load(f)
    return model

def load_vectorizer(vectorizer_name=vectorizer_name):
    # Load Vectorizer
    with open(vectorizer_name, 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer


def train():
    # # SENTIMENT 140#####################
    data = pd.read_csv(file_location, nrows=max_entries, skiprows=1, header=None)
    # Review Text
    review_column = data[0]
    # Just Text Data
    review_text = []
    for review in review_column:
        review_text.append(str(review))
    
    # Data Analysis
    review_text_vectorized = vectorize_data(review_text)
    # Just 
    labels = []
    label_column = data[1]
    for label in label_column:
        if label == 4:
            labels.append(1)
        else:
            labels.append(0)
    ######################################
    print(f'LABELS Length: {len(labels)}')
    print("Vectorized SHape")
    print(review_text_vectorized.shape)

    # Train Classifier and Test Accuracy
    classifier = train_classifier(review_text_vectorized, labels)
    # Save Classifier
    save_model(classifier)

def twitter_vaildate():
    # Load 
    pass

def validate():
    prediction_engine = PredictionEngine()

    # Load CSV File, but strip it first
    data = pd.read_csv(validation_csv_location, header=None, skiprows=1, nrows=validation_rows)
    # Remove first column
    data = data.drop(columns=0)
    # Remove \n from every entry
    data = data.applymap(lambda x: x.replace('\n', ''))
    
    # Put in Lists
    better_sentiment = data.iloc[:, 0].tolist()
    worse_sentiment = data.iloc[:, 1].tolist()

    better_sentiment_predictions = []
    worse_sentiment_predictions = []

    counter = 0
    for sentiment in better_sentiment:
        better_sentiment_predictions.append(prediction_engine.predict(sentiment))
        counter += 1
        if counter % 10 == 0:
            print(f'{counter} Predictions Made')
    counter = 0
    for sentiment in worse_sentiment:
        worse_sentiment_predictions.append(prediction_engine.predict(sentiment))
        counter += 1
        if counter % 10 == 0:
            print(f'{counter} Predictions Made')

    print(f'Unable to Predict {prediction_engine.unable_to_predict_total}')

    # Compare Relative Accuracy
    correct = 0
    total = 0
    for better, worse in zip(better_sentiment_predictions, worse_sentiment_predictions):
        # Good = 1, Bad = 0
        if better>worse:
            correct += 1
        if better != worse:
            total += 1
    
    print(f'Correct: {correct}')
    print(f'Total: {total}')
    print(f'Accuracy: {correct/total}')

class PredictionEngine:

    # NEGATIONS LIST
    negations_list = ['aint', 'arent', 'cannot', 'cant', 'couldnt', 'darent', 'didnt', 'doesnt', 
    'ain\'t', 'aren\'t', 'can\'t', 'couldn\'t', 'daren\'t', 'didn\'t', 'doesn\'t', 
    'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neither', 
    'don\'t', 'hadn\'t', 'hasn\'t', 'haven\'t', 'isn\'t', 'mightn\'t', 'mustn\'t', 
    'neednt', 'needn\'t', 'never', 'none', 'nope', 'nor', 'not', 'nothing', 'nowhere', 
    'oughtnt', 'shant', 'shouldnt', 'uhuh', 'wasnt', 'werent', 'oughtn\'t', 'shan\'t', 
    'shouldn\'t', 'uh-uh', 'wasn\'t', 'weren\'t', 'without', 'wont', 'wouldnt', 'won\'t', 
    'wouldn\'t', 'rarely', 'seldom', 'despite', 'jk']#, 'but', 'however', 'yet', 'although']

    # Positive Threshold
    positive_threshold = 0.6
    # Negative Threshold
    negative_threshold = 0.4

    # 0.5, 0.5 = 66%, 0 unable to predict = 66%
    # 0.6, 0.4 = 65%, 4 unable to predict = 69%
    # 0.7, 0.3 = 58%, 20 unable to predict = 78%

    def __init__(self, vectorizer = load_vectorizer(), classifier = load_model()):
        # Initialize Vectorizer and Classifier
        self.vectorizer = vectorizer
        self.classifier = classifier

        # Unable to Predict Counter
        self.unable_to_predict_total = 0
    
    # Returns Positive Probability
    def predict(self, string):
        # Predicts Word By Word Basis (Including Negations)
        positive_probability = self.predict_with_negations(string)
        # Positive Probability (0 = Negative, 1 = Positive)
        return positive_probability

    def predict_with_negations(self, string):
        # Split String
        string_split = string.split()
        # Get Locations of Negations
        negation_locations = np.array([i for i, word in enumerate(string_split) if word in self.negations_list])

        # Simplify Stringed Negations (Returns what indices to remove)
        def simplify_negation_locations(negation_locations):

            # Negations to Remove (Return)
            negations_to_remove = []

            # Get Sequences
            sequences = []
            for location in negation_locations:
                if len(sequences) == 0:
                    sequences.append([location])
                elif location - sequences[-1][-1] == 1:
                    sequences[-1].append(location)
                else:
                    sequences.append([location])

            # Find Even Negations to Remove (Cancel Out)
            # Shorten Odd Negations to One (Simplify)
            for sequence in sequences:
                if len(sequence) % 2 == 0:
                    negations_to_remove.append(sequence)
                elif len(sequence) % 2 == 1:
                    negations_to_remove.append(sequence[1:])
            negations_to_remove = [item for sublist in negations_to_remove for item in sublist]

            # Return Simplified list of negation_locations
            return negations_to_remove

        # Simplify
        negation_locations_to_remove = simplify_negation_locations(negation_locations)
        
        # remove redundant negations
        new_string_list = []
        for i in range(len(string_split)):
            if i not in negation_locations_to_remove:
                new_string_list.append(string_split[i])
        string_split = new_string_list

        # Updated Negation Locations
        negation_locations = np.array([i for i, word in enumerate(string_split) if word in self.negations_list])

        # Find Notably Positive and Negative Words (Excluding Negations) [word, value, index_in_phrase]
        notable_words_plus_value = []

        words_no_negations = [i for i in string_split if i not in self.negations_list]
        for word in words_no_negations:
            # Vectorize Data
            vectorized_data = self.vectorizer.transform([word])
            # Predict
            positive_prediction = self.classifier.predict_proba(vectorized_data)[0]
            if positive_prediction[1] > self.positive_threshold:
                notable_words_plus_value.append([word, positive_prediction[1], string_split.index(word)])
            elif positive_prediction[1] < self.negative_threshold:
                notable_words_plus_value.append([word, positive_prediction[1], string_split.index(word)])

        # If Notable Word Has Negation Before It or After It (Use Get Opposite of Value)
        notable_word_values = []
        for word in notable_words_plus_value:
            negations_found = 0
            # If Word Before is a Negation
            if word[2] != 0 and string_split[word[2]-1] in self.negations_list: 
                negations_found += 1
            # If Word 2 Before is a Negation
            elif word[2] != 1 and string_split[word[2]-2] in self.negations_list:
                negations_found += 1
            # If Word is After a Negation
            if not word[2] > len(string_split)-2 and string_split[word[2]+1] in self.negations_list:
                negations_found += 1

            # # If Word 2 After is a Negation -> Very Rare (May Not Even Occur in English)
            # elif not word[2] > len(string_split)-3 and string_split[word[2]+2] in self.negations_list:
            #     negations_found += 1

            # Otherwise, not Negated
            if negations_found % 2 == 0:
                notable_word_values.append([word[0], word[1], word[2]])
            elif negations_found % 2 == 1:
                notable_word_values.append([word[0], abs(1-word[1]), word[2]])
        
        # print(f'Notable Words sdf: {notable_word_values}')

        # Returns Average of Notable Word Values
        if len(notable_word_values) is not 0:
            value_list = []
            for word in notable_word_values:
                value_list.append(word[1])
            return np.average(np.array(value_list))
        else:
            self.unable_to_predict_total += 1
            return 0.5


if __name__ == '__main__':

    # prediction = PredictionEngine().predict("")
    # print(prediction)

    # model = load_model()
    # vectorizer = load_vectorizer()
    # print(model.predict_proba(vectorizer.transform(['this is not cool. this is not awesome'])))

    # train()
    validate()