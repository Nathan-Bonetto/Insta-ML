import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from sklearn.feature_extraction.text import CountVectorizer
import math

def main():
    # Download stopwords
    nltk.download('stopwords')

    # Import "old" csv file and randomize it before splitting
    data = pd.read_csv("project_data.csv")
    df_data = data.reindex(np.random.permutation(data.index))

    # Import new data to test against
    new_data = pd.read_csv("postData.csv")
    new_data.drop(new_data.columns[0], 1, inplace=True)

    # Separate training data from validating data
    total = len(df_data)
    percent = .8
    training_examples = math.ceil(total * percent)
    validation_examples = total - training_examples

    # Data to train with
    training_data = df_data["Message"].head(training_examples)
    training_target = df_data["Spam"].head(training_examples)

    # Data to test the training
    test_data = df_data["Message"].tail(validation_examples)
    test_target = df_data["Spam"].tail(validation_examples)

    # Brand-new data that does not have validation
    new_test = new_data["Message"]

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import SGDClassifier

    # Pipeline stores all functions needed to pass Strings into the ML model
    text_clf = Pipeline([('vect', CountVectorizer(analyzer=clean_text)), ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None))])

    # Fix data into model for training
    text_clf.fit(training_data, training_target)

    # Make prediction from training data
    pred = text_clf.predict(test_data)

    # Print and test training data
    print(np.mean(pred == test_target))

    # Display training data
    from sklearn import metrics

    print(metrics.classification_report(test_target, pred))

    print(metrics.confusion_matrix(test_target, pred))

    # Make prediction on brand-new data
    pred = text_clf.predict(new_test)
    print(pred)

    # Push newly predicted data to csv file for manual approval
    store_to_csv(new_data, pred)

def store_to_csv(data, pred):
    output = data.copy()
    output["Spam"] = pred
    output.to_csv("ML_Prediction.csv")

def clean_text(input_pd):
    # Weed out punctuations
    temp_pd = [x for x in input_pd if x not in string.punctuation]
    temp_pd = ''.join(temp_pd)

    # Weed out stopwords
    new_pd = [x for x in temp_pd.split() if x.lower() not in stopwords.words('english')]

    return new_pd

main()