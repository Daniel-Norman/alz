_TODO: more detailed instructions about how to run both_

Performs machine learning on the data provided by the LBP feature extraction program.
This script learns to predict a patient's presence of Alzheimer's Disease.

### AdaBoost + Extra Trees
Uses [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) with [Extra Trees](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html) base classifiers.
Performs cross validation using a [Stratified Shuffle Split](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) with 30 folds and 5 samples per fold. We use [F1 Score](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html) to rate each classifier.

Due to the random nature of the classifiers, we repeatedly generate a new random seed, build a classifier using the seed, and evaluate it with cross evaluation. The best performing classifier is kept and tested one more time with 100-fold 5-sample cross validation.

Data directory should contain two files for each patient: their histogram from LBP, and their lesion volume from lesion detection.
Example: patient 1234 should contribute `lbp_histogram_1234.csv` and `lesions_volume_1234.csv` to the directory.

Also requires a patient-to-Alz CSV to indicate the presence of Alzheimer's for each patient (you must create this yourself).

Example for patients 555 and 666 with AD and patient 444 without:
```
555,1
444,0
666,1
``` 



Run using

`python trees.py [number of iterations] [label csv] [data directory]`

### CNN
Uses a convolutional neural network from [TensorFlow](https://www.tensorflow.org/).

Also requires a filename-to-label CSV to indicate the cognitive impairment of each histogram's source patient.
Example for patients 1 and 3 with cognitive impairment and patient 2 without:
```
flair1.nii,1
flair2.nii,0
flair3.nii,1
``` 

Run using

`python cnn.py [training label CSV] [training histogram directory] [test label CSV] [test histogram directory]`

Outputs the accuracy of the model.
