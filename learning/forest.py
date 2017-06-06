import csv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import entropy

if len(sys.argv) != 4:
    print 'Expects 3 arguments: label_csv training_directory test_directory'
    quit()

print 'Running Random Forest learner on histogram data.'
label_csv = sys.argv[1]
training_directory = sys.argv[2]
test_directory = sys.argv[3]

print 'Training directory: %s' % training_directory
print 'Test directory: %s' % test_directory


training_data = []
training_labels = []
test_data = []
test_labels = []


# Load the label associated with each image
def setup_labels(label_file):
    labels_per_hist = {}
    with open(label_file) as lbl_csv:
        reader = csv.reader(lbl_csv)
        for row in reader:
            labels_per_hist['lbp_histogram_%s.csv' % row[0]] = int(row[1])
    return labels_per_hist


def read_data(data, labels, labels_per_hist, directory):
    csvs = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.csv')]

    for f in csvs:
        file_path = join(directory, f)
        with open(file_path, 'rb') as histogram_file:
            reader = csv.reader(histogram_file)
            hist = []
            for value in reader.next():
                hist.append(float(value))
            hist = np.array(hist)
            data.append(process_histogram(hist))
            labels.append(labels_per_hist[f])


# Process the LBP histogram to convert it to the features used in training
def process_histogram(hist):
    # Generate data that could have generated this histogram, to be used as a possible feature
    built_data = []
    for i, value in enumerate(hist):
        for _ in xrange(int(value*1000)):
            built_data.append(i)

    features = [
        np.std(built_data), np.mean(built_data), np.median(built_data), entropy(built_data),
        np.std(hist), np.mean(hist), np.median(hist), entropy(hist),
    ]
    return features

print 'Setting up labels...'
labels_per_histogram = setup_labels(label_csv)

print 'Loading training data...'
read_data(training_data, training_labels, labels_per_histogram, training_directory)
print '\tLoaded %s training samples.' % len(training_labels)

print 'Loading test data...'
read_data(test_data, test_labels, labels_per_histogram, test_directory)
print '\tLoaded %s test samples.' % len(test_labels)


print 'Training the random forest...'
clf = RandomForestClassifier(n_estimators=20)
clf.fit(training_data, training_labels)

print '\nPerformance of classifier on test data:'
predicted_labels = clf.predict(test_data)
print classification_report(test_labels, predicted_labels, target_names=['No CI', 'CI'])
