import csv
import numpy as np
import sys
import random
from os import listdir
from os.path import isfile, join
from scipy.stats import entropy
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score


if len(sys.argv) != 4:
    print 'Expects 3 arguments: n_iterations label_csv histogram_directory'
    quit()

HISTOGRAM_PREFIX = 'lbp_histogram_'
VOLUME_PREFIX = 'lesions_volume_'


# Load the label associated with each image
def setup_labels(label_file):
    labels_per_hist = {}
    with open(label_file) as lbl_csv:
        reader = csv.reader(lbl_csv)
        for row in reader:
            labels_per_hist[row[0]] = int(row[1])
    return labels_per_hist


def read_data(labels_per_file, directory):
    data = []
    labels = []

    hist_csvs = [f for f in listdir(directory) if isfile(join(directory, f)) and f.endswith('.csv') and "volume" not in f]

    for f in hist_csvs:
        file_path = join(directory, f)
        with open(file_path, 'rb') as histogram_file:
            hist_reader = csv.reader(histogram_file)
            hist = []
            for value in hist_reader.next():
                hist.append(float(value))
            hist = np.array(hist)
            sample_number = f.replace(HISTOGRAM_PREFIX, '').replace('.csv', '')
            with open(join(directory, '%s%s.csv' % (VOLUME_PREFIX, sample_number)), 'rb') as volume_file:
                volume_reader = csv.reader(volume_file)
                volume = float(volume_reader.next()[0])

                data.append(process_histogram(hist, volume))
                labels.append(labels_per_file[sample_number])

    return data, labels


# Process the LBP histogram to convert it to the features used in training
def process_histogram(hist, volume):
    # Generate data that could have generated this histogram, to be used as a possible feature
    built_data = []
    for i, value in enumerate(hist):
        for _ in xrange(int(value*1000)):
            built_data.append(i)

    features = [
        volume,
        np.std(built_data), np.mean(built_data), np.median(built_data), entropy(built_data),
        np.std(hist), np.mean(hist), np.median(hist), entropy(hist),
    ]

    combined_features = []
    for feature_a in features:
        for feature_b in features:
            combined_features.append(feature_a * feature_b)

    return combined_features


def train_model(n_iterations, data, labels):
    print 'Training an AdaBoost(RandomForest) model %s times with 30-split cross validation...' % n_iterations
    max_f1 = 0.0
    max_f1_std = 0.0
    max_clf = None
    printing_interval = max(1, min(n_iterations/10, 10))
    for i in xrange(n_iterations):
        # Use AdaBoost to combine base classifiers, weighing incorrectly-predicted samples heavily as time goes on.
        # Use an ExtraTrees classifier, to build decision trees that select on some set of features that seem to
        # best describe the data.
        clf = AdaBoostClassifier(
            base_estimator=ExtraTreesClassifier(random_state=random.randint(0, 2**32)),
            random_state=random.randint(0, 2**32),
            n_estimators=10
        )
        # Use cross validation to assess the performance of this classifier, needed because of the small data set.
        # Cross validation allows us to validate the performance against multiple groupings of test data, all while
        # maintaining the requirement that a model is never evaluated on data is has seen during training.
        # Use a Stratified Shuffle Split to randomly chose 5 points as test data while trying to maintain an equal
        # proportion of test samples with and without Alzheimer's. Perform this CV 30 times.
        scores = cross_val_score(clf, data, labels, cv=StratifiedShuffleSplit(test_size=5, n_splits=30), scoring='f1_macro')
        # Report F1 score, as it is a better overall measure of performance compared to just accuracy
        average = scores.mean()
        std = scores.std()
        if average - 2*std > max_f1 - 2*max_f1_std:
            max_f1 = average
            max_clf = clf
            max_f1_std = std
        if i % printing_interval == 0:
            print 'Done with iteration %i. Best model so far has F1= %.4f (+/- %.4f)' % (i, max_f1, 2*max_f1_std)
    return max_clf, max_f1, max_f1_std


print 'Setting up labels...'
label_map = setup_labels(label_file=sys.argv[2])
print 'Loading data...'
data, labels = read_data(labels_per_file=label_map, directory=sys.argv[3])
print 'Loaded %s samples.' % len(data)

best_clf, best_f1, best_f1_std = train_model(n_iterations=int(sys.argv[1]), data=data, labels=labels)
print 'Max F1 score achieved: %.4f (+/- %.4f)' % (best_f1, 2*best_f1_std)
print 'Using model:\n%s' % best_clf
print 'Evaluating model one last time using 100-split cross validation...'
scores = cross_val_score(best_clf, data, labels, cv=StratifiedShuffleSplit(test_size=5, n_splits=100), scoring='f1_macro')
print 'F1: %.4f (+/- %.4f)' % (scores.mean(), 2*scores.std())