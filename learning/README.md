Performs machine learning on the data provided by the LBP feature extraction program.
This script learns to predict a patient's cognitive impairment (CI).

Currently using a [Random Forest classifier from sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Run using

`python learner.py [training data directory] [test data directory] [histogram size]`

Outputs a [classification report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
of the results.
