Linear binary pattern feature extraction process.
Run on the original MRI scan, but only on the provided regions-of-interest given by the CSV of lesion regions.

Run using

`lbp_batch.sh [# in parallel] [MRI folder] [lesion csv folder] [output histogram folder]`

for batch or

`python lbp.py [MRI file] [lesion regions CSV] [output histogram CSV]`

for a single file.

The output is a single normalized histogram of uniform LBP buckets for the lesion regions.
