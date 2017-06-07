_TODO: more detailed instructions about how to run both_

Performs machine learning on the data provided by the LBP feature extraction program.
This script learns to predict a patient's cognitive impairment (CI).

### Random Forest
Uses a [Random Forest classifier from sklearn](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

_TODO: update this to latest version description, talking about cross validation etc._

Requires a patient-to-Alz CSV to indicate the presence of Alzheimer's for each patient.

Example for patients 555 and 666 with AD and patient 444 without:
```
555,1
444,0
666,1
``` 

Run using

`python forest.py [number of iterations] [label csv] [data directory]`

Outputs the 95% confidence range of the F1 Score of the results from cross validation using the best classifier found after iterating.

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
