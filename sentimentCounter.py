import csv

negative = 0
neutral = 0
positive = 0

negative_tweets = []
neutral_tweets = []
positive_tweets = []

time_elapsed = 0
accuracy = 0
report = None

file_to_read = 'predictions/predictions_svm_count_1000000_rowsRead.csv'
with open(file_to_read) as sentiment_file:
    csv_reader = csv.reader(sentiment_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            line_count += 1
            pass
        elif line_count == 1:
            time_elapsed = row[0]
            accuracy = row[1]
            report = row[2]
            line_count += 1
        elif line_count < 4:
            line_count += 1
            pass
        else:
            if row[0] == "0":
                negative += 1
                negative_tweets.append(row[1])
            elif row[0] == "2":
                neutral += 1
                neutral_tweets.append(row[1])
            elif row[0] == "4":
                positive += 1
                positive_tweets.append(row[1])
            line_count += 1

print("\nnegative: ", negative, " neutral: ", neutral, " positive: ", positive)
print("\nCLASSIFIER STATS:")
print("Time elapsed: ", time_elapsed)
print("Report: ", report)
print(accuracy)
