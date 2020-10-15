import pandas as pd
from Preprocessing import cleaner
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report, accuracy_score, plot_confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
import matplotlib.pyplot as plt
import time
import sys
import csv

rows_to_read = 10000
rows_to_read_set = False
while not rows_to_read_set:
    try:
        prompt_rows_to_read = int(input("How many rows do you want to read from the data set?"))
        if prompt_rows_to_read > 0:
            rows_to_read = prompt_rows_to_read
            rows_to_read_set = True
        else:
            print("Please type a number larger than 0\n")

    except ValueError as e:
        print("Please type a number \n")

"""
IF WE WANT TO USE SEPARATE DATA SETS FOR TRAINING AND TESTING

# Read training data from train.csv and save score and text separately as X_train and y_train
training_file = 'tweets_with_sentiment.csv'
df_train = pd.read_csv(training_file, encoding='utf-8', nrows=rows_to_read)
X_train = cleaner(df_train).text
y_train = df_train.score

# Read test data from testNeutral.csv and save score and text separately as X_Test and y_test
test_file = 'testNeutral.csv'
df_test = pd.read_csv(test_file, encoding='utf-8')
X_test = cleaner(df_test).text
y_test = df_test.score
"""

# Read training data from training_data_set and save score and text separately as X_train and y_train
training_data_set = 'tweets_with_sentiment.csv'
df_train = pd.read_csv(training_data_set, encoding='utf-8', nrows=rows_to_read)
all_X_data = cleaner(df_train).text
all_y_data = df_train.score

X_train, X_test, y_train, y_test = train_test_split(all_X_data, all_y_data, test_size=0.8, random_state=0)

# Read evaluation data from evaluation_data_set
evaluation_data_set = 'Clean_Tweets_no_similarities.csv'
df_eval = pd.read_csv(evaluation_data_set, encoding='utf-8')
X_eval = cleaner(df_eval).text


def choose_classifier():
    while True:
        classifier_choice = input("Which classifier do you want to use? (nb, svm or neural)").lower()

        if classifier_choice == "q" or classifier_choice == "quit":
            sys.exit("Program ended by user")
        elif classifier_choice == "nb":
            return classifier_choice, MultinomialNB()
        elif classifier_choice == "svm":
            return classifier_choice, svm.SVC(kernel='linear', C=1.0, cache_size=8000, verbose=10)
        elif classifier_choice == "neural":
            return classifier_choice, MLPClassifier(max_iter=100, alpha=0.0001,
                                                    solver='adam', verbose=True, random_state=21, tol=0.0001)


def choose_vectorizer():
    while True:
        vectorizer_choice = input("Which vectorizer do you want to use? (count or tfidf)").lower()

        if vectorizer_choice == "count":
            return vectorizer_choice, CountVectorizer(binary=True)
        elif vectorizer_choice == "tfidf":
            return vectorizer_choice, TfidfVectorizer(min_df=5, max_df=0.8, ngram_range=(1, 2))
        elif vectorizer_choice == "q" or vectorizer_choice == "quit":
            sys.exit("Program ended by user")


def plot_result(X_Test_vect):
    titles_options = [("\nConfusion matrix, without normalization", None),
                      ("\nNormalized confusion matrix", 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(classifier, X_Test_vect, y_test, normalize=normalize)
        disp.ax_.set_title(title)
        # print(title)
        # print(disp.confusion_matrix)

    plt.show()
    plt.close('all')


def train_predict_and_plot(classifier, classifier_type, vectorizer, vectorizer_type):
    start_time = time.time()
    """
    
    CROSS VALIDATION NO LONGER USED 
    # ========================= CROSS VALIDATION ===========================
    print("\n************************ CROSS VALIDATION **************************")
    X_Cross_data = vectorizer.fit_transform(all_X_data)
    scores = cross_val_score(classifier, X_Cross_data, all_y_data, cv=5, verbose=10)
    print("Scores: ", scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("************************ END OF CROSS VALIDATION *******************\n")
    """

    # ========================= TRAINING MODEL AND PREDICTING ===========================
    train_vectors = vectorizer.fit_transform(X_train)
    classifier.fit(train_vectors, y_train)
    X_Test_vect = vectorizer.transform(X_test)
    prediction = classifier.predict(X_Test_vect)

    report = classification_report(y_test, prediction, output_dict=True)

    end = time.time()
    time_elapsed = round(end - start_time, 2)
    print(classifier_type + "-classifier using " + vectorizer_type + "-vectorizer took: ", time_elapsed,
          " seconds\n")
    test_accuracy = "Accuracy: {:.2f}%".format(accuracy_score(y_test, prediction) * 100)
    print(test_accuracy)
    print(vectorizer_type + "-vectorizer negative: ", report["0"])
    print(vectorizer_type + "-vectorizer neutral: ", report["2"])
    print(vectorizer_type + "-vectorizer positive: ", report["4"])

    # print("\nF1 Score: {:.2f}".format(f1_score(y_test, y_pred) * 100))

    # ===================== REAL TWEET DATA =================================
    X_eval_vect = vectorizer.transform(X_eval)
    real_prediction = classifier.predict(X_eval_vect)
    filename = "predictions/predictions_" + chosen_classifier + "_" + chosen_vectorizer + "_" + str(
        rows_to_read) + "_rowsRead.csv"
    with open(filename, mode='w') as prediction_file:
        prediction_writer = csv.writer(prediction_file, delimiter=',')

        prediction_writer.writerow((['time_elapsed', "accuracy", "report"]))
        prediction_writer.writerow([str(time_elapsed) + "s", test_accuracy, report])
        prediction_writer.writerow(["\n"])
        prediction_writer.writerow(['Sentiment', 'Text'])
        for index, pred in enumerate(real_prediction):
            prediction_writer.writerow([pred, df_eval.text[index]])
        print("\nWrote predicted results to file")
    # =======================================================================

    plot_answered = False
    while not plot_answered:
        prompt_plot_result = input("\nDo you want to plot? (y,n)").lower()
        if prompt_plot_result == "y" or prompt_plot_result == "yes":
            plot_result(X_Test_vect)
            plot_answered = True
        elif prompt_plot_result == "n" or prompt_plot_result == "no":
            return


# RUN PROGRAM
print("\n ================= FILES ARE READ AND STORED ===================\n")

chosen_classifier, classifier = choose_classifier()
chosen_vectorizer, vectorizer = choose_vectorizer()
print("Running test with Classifier: ", chosen_classifier, " and Vectorizer: ", chosen_vectorizer)
train_predict_and_plot(classifier, chosen_classifier, vectorizer, chosen_vectorizer)
