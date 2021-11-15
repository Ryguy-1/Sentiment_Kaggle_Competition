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
validation_rows = 5000

# Parameters
file_location = "Sentiment_140/sentiment_train_cleaned.csv"
max_entries = 1599996
# Models
model_vectorizer_number = 2

# Paths for Model and Vectorizer
model_name = f'Model_{model_vectorizer_number}/model_{model_vectorizer_number}.pkl'
vectorizer_name = f'Model_{model_vectorizer_number}/vectorizer_{model_vectorizer_number}.pkl'



# NEGATIONS LIST
negations_list = ['aint', 'arent', 'cannot', 'cant', 'couldnt', 'darent', 'didnt', 'doesnt', 
'ain\'t', 'aren\'t', 'can\'t', 'couldn\'t', 'daren\'t', 'didn\'t', 'doesn\'t', 
'dont', 'hadnt', 'hasnt', 'havent', 'isnt', 'mightnt', 'mustnt', 'neither', 
'don\'t', 'hadn\'t', 'hasn\'t', 'haven\'t', 'isn\'t', 'mightn\'t', 'mustn\'t', 
'neednt', 'needn\'t', 'never', 'none', 'nope', 'nor', 'not', 'nothing', 'nowhere', 
'oughtnt', 'shant', 'shouldnt', 'uhuh', 'wasnt', 'werent', 'oughtn\'t', 'shan\'t', 
'shouldn\'t', 'uh-uh', 'wasn\'t', 'weren\'t', 'without', 'wont', 'wouldnt', 'won\'t', 
'wouldn\'t', 'rarely', 'seldom', 'despite', 'jk']#, 'but', 'however', 'yet', 'although']


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
    # Load CSV File, but strip it first
    data = pd.read_csv(validation_csv_location, header=None, skiprows=1, nrows=validation_rows)
    # Remove first column
    data = data.drop(columns=0)
    # Remove \n from every entry
    data = data.applymap(lambda x: x.replace('\n', ''))
    
    # Put in Lists
    better_sentiment = data.iloc[:, 0].tolist()
    worse_sentiment = data.iloc[:, 1].tolist()

    print(f'Better Sentiment : {better_sentiment[0:10]}')

    # Load Vectorizer
    vectorizer = load_vectorizer()
    # Load Classifier
    classifier = load_model()

    # Vectorize Data
    better_sentiment_vectorized = vectorizer.transform(better_sentiment)
    worse_sentiment_vectorized = vectorizer.transform(worse_sentiment)

    print(better_sentiment_vectorized.shape)

    # Predict
    better_sentiment_predictions = classifier.predict_proba(better_sentiment_vectorized)
    worse_sentiment_predictions = classifier.predict_proba(worse_sentiment_vectorized) 

    # Print Predictions
    print(f'Better Sentiment Predictions: {better_sentiment_predictions[0][1]}')
    print(f'Worse Sentiment Predictions: {worse_sentiment_predictions[0][1]}')

    # Compare Relative Accuracy
    correct = 0
    total = 0
    for better, worse in zip(better_sentiment_predictions, worse_sentiment_predictions):
        # Good = 1, Bad = 0
        better = better[1]
        worse = worse[1]
        if better>worse:
            correct += 1
        if better != worse:
            total += 1
    
    print(f'Correct: {correct}')
    print(f'Total: {total}')
    print(f'Accuracy: {correct/total}')

if __name__ == '__main__':
    validate()
    # test_string = ""
    # # Test if there is a negation
    # negations_found = 0
    # test_string_list = test_string.split()
    # for word in negations_list:
    #     for word2 in test_string_list:
    #         if word2.lower() == word.lower():
    #             negations_found += 1
    
    # positivity = test_individual_sentiments(load_model(), load_vectorizer(), [test_string])
    # print(f'Positivity: {positivity}')

    # if negations_found > 0:
    #     print(f'Negation Found: {negations_found}')
    #     print(f'positivity: {positivity}')
    #     for i in range(negations_found):
    #         positivity = abs(1 - positivity)
    #         print(f'positivity: {positivity}')
    #     print(f'Positivity: {positivity}')
    # else:
    #     print(f'Positivity: {positivity}')


    # train()