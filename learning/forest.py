import csv
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from scipy.stats import entropy
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


if len(sys.argv) != 4:
    print 'Expects 3 arguments: n_iterations label_csv histogram_directory'
    quit()


# Load the label associated with each image
def setup_labels(label_file):
    labels_per_hist = {}
    with open(label_file) as lbl_csv:
        reader = csv.reader(lbl_csv)
        for row in reader:
            labels_per_hist['lbp_histogram_%s.csv' % row[0]] = int(row[1])
    return labels_per_hist


def read_data(labels_per_hist, directory):
    data = []
    labels = []

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

    return data, labels


# Process the LBP histogram to convert it to the features used in training
def process_histogram(hist):
    # Generate data that could have generated this histogram, to be used as a possible feature
    built_data = []
    for i, value in enumerate(hist):
        for _ in xrange(int(value*1000)):
            built_data.append(i)

    features = [
        # TODO: entropy
        # TODO: other stuff that doesn't come from histogram, like (total lesion volume / image volume)?
        np.std(built_data), np.mean(built_data), np.median(built_data), entropy(built_data),
        np.std(hist), np.mean(hist), np.median(hist), entropy(hist),
    ]
    return features


def train_model(n_iterations, data, labels):
    print 'Training an AdaBoost(RandomForest) model %s times...' % n_iterations
    max_f1 = 0.0
    max_clf = None
    for i in xrange(n_iterations):
        # Use AdaBoost to combine base classifiers, weighing incorrectly-predicted samples heavily as time goes on.
        # Use a RandomForest classifier, to build decision trees that select on some set of features that seem to
        # best describe the data.
        clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=10)
        # Use cross validation to assess the performance of this classifier, needed because of the small data set.
        # Cross validation allows us to validate the performance against multiple groupings of test data, all while
        # maintaining the requirement that a model is never evaluated on data is has seen during training.
        # Use a Stratified Shuffle Split to randomly chose your test data while trying to maintain an equal proportion
        # of test samples with and without Alzheimer's.
        scores = cross_val_score(clf, data, labels, cv=StratifiedShuffleSplit(test_size=5), scoring='f1_macro')
        # Report F1 score, as it is a better overall measure of performance compared to just accuracy
        max_f1 = max(scores.mean(), max_f1)
        average_f1 = scores.mean()
        if average_f1 > max_f1:
            max_f1 = average_f1
            max_clf = clf
        if i % (n_iterations/10) == 0:
            print 'Done with iteration %i. Best model so far has F1=%s' % (i, max_f1)
    return max_clf, max_f1


print 'Setting up labels...'
label_map = setup_labels(label_file=sys.argv[2])
print 'Loading data...'
data, labels = read_data(labels_per_hist=label_map, directory=sys.argv[3])
print 'Loaded %s samples.' % len(data)

best_clf, best_f1 = train_model(n_iterations=int(sys.argv[1]), data=data, labels=labels)
print 'Max F1 score achieved: %s' % best_f1
