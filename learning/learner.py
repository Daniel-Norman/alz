import csv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

if len(sys.argv) != 4:
    print 'Expects 3 arguments: training_directory test_directory histogram_size'
    quit()

print 'Running Random Forest learner on histogram data.'
print 'Training directory: %s' % sys.argv[1]
print 'Test directory: %s' % sys.argv[2]
histogram_size = int(sys.argv[3])


training_data = []
training_labels = []
test_data = []
test_labels = []

print 'Loading training data...'
training_csvs = [join(sys.argv[1],f) for f in listdir(sys.argv[1]) if isfile(join(sys.argv[1], f))]
for f in training_csvs:
    with open(f, 'rb') as training_csv:
        reader = csv.reader(training_csv)
        label = int(reader.next()[0])
        if label != 0:
            label = 1
        hist = np.empty(histogram_size)
        for i, value in enumerate(reader.next()):
            hist[i] = (int(value))
        training_data.append(hist)
        training_labels.append(label)
print '\tLoaded %s training samples.' % len(training_labels)

print 'Loading test data...'
test_csvs = [join(sys.argv[2],f) for f in listdir(sys.argv[2]) if isfile(join(sys.argv[2], f))]
for f in test_csvs:
    with open(f, 'rb') as test_csv:
        reader = csv.reader(test_csv)
        label = int(reader.next()[0])
        if label != 0:
            label = 1
        hist = np.empty(histogram_size)
        for i, value in enumerate(reader.next()):
            hist[i] = (int(value))
        test_data.append(hist)
        test_labels.append(label)
print '\tLoaded %s test samples.' % len(test_labels)

print 'Training the random forest...'
clf = RandomForestClassifier()
clf.fit(training_data, training_labels)

print '\nPerformance of classifier on test data:'
predicted_labels = clf.predict(test_data)
print classification_report(test_labels, predicted_labels, target_names=['No CI', 'CI'])
