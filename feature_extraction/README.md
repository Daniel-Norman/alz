Linear binary pattern feature extraction process.
Run on the original MRI scan, but only on the provided regions-of-interest given by the CSV of lesion regions.

`python lbp.py [MRI file] [lesion regions CSV] [output histogram CSV]`

In order to batch all of the lbp files, run the following with lbp.py in the
same directory. It will save all of the histograms in the output folder as 'lbp_histogram_[number].csv'

`./lbp_batch.sh [# in parallel] [MRI folder] [lesion csv folder] [output histogram folder]

TODO: fix this description: The output should be a texture and pattern recognition file that can be viewed
within BrainSuite.
