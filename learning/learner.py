import csv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if len(sys.argv) != 5:
    print 'Expects 4 arguments: training_label_csv training_directory test_label_csv test_directory'
    quit()

print 'Running Random Forest learner on histogram data.'
training_label_csv = sys.argv[1]
training_directory = sys.argv[2]
test_label_csv = sys.argv[3]
test_directory = sys.argv[4]

print 'Training directory: %s' % training_directory
print 'Test directory: %s' % test_directory


training_data = []
training_labels = []
test_data = []
test_labels = []


def read_data(data, labels, label_file, directory):
    data_index_map = {}
    csvs = [f for f in listdir(directory) if isfile(join(directory, f))]
    index = 0
    for f in csvs:
        file_path = join(directory, f)
        with open(file_path, 'rb') as histogram_file:
            reader = csv.reader(histogram_file)
            hist = []
            for i, value in enumerate(reader.next()):
                hist.append(float(value))
            hist = np.array(hist)
            data.append(process_histogram(hist))
            labels.append(0)
            data_index_map[f] = index
            index += 1
    with open(label_file) as label_csv:
        reader = csv.reader(label_csv)
        for row in reader:
            training_labels[data_index_map[row[0]]] = int(row[1])


# Process the LBP histogram to convert it to the features used in training
def process_histogram(hist):
    # Right now, just use it directly
    # TODO: look into converting to statistical measurements like mean/entropy/... like the paper does
    return hist

print 'Loading training data...'
read_data(training_data, training_labels, training_label_csv, training_directory)
print '\tLoaded %s training samples.' % len(training_labels)

print 'Loading test data...'
read_data(test_data, test_labels, test_label_csv, test_directory)
print '\tLoaded %s test samples.' % len(test_labels)


print 'Training the random forest...'
clf = RandomForestClassifier()
clf.fit(training_data, training_labels)

print '\nPerformance of classifier on test data:'
predicted_labels = clf.predict(test_data)
print classification_report(test_labels, predicted_labels, target_names=['No CI', 'CI'])
