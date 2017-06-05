_TODO: more detailed instructions about how to run both_

Performs machine learning on the data provided by the LBP feature extraction program.
This script learns to predict a patient's cognitive impairment (CI).

### Random Forest
Uses a [Random Forest classifier from sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

Requires a filename-to-label CSV to indicate the cognitive impairment of each histogram's source patient.

Example for patients 1 and 3 with cognitive impairment and patient 2 without:
```
histogram1.csv,1
histogram2.csv,0
histogram3.csv,1
``` 

Run using

`python forest.py [training label CSV] [training histogram directory] [test label CSV] [test histogram directory]`

Outputs a [classification report](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
of the results.

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
