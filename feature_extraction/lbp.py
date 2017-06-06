import csv
import sys
import nibabel as nib
import numpy as np
from skimage import feature

NUM_POINTS = 8
RADIUS = 2
# Expand the bounding box so we include a little bit of region around the lesion when performing LBP
EXPAND_PIXELS = RADIUS*2

if len(sys.argv) != 4:
    print 'Expects 3 arguments: mri_file lesion_csv output_csv'
    quit()

mri_file = sys.argv[1]
lesion_csv_file = sys.argv[2]
output_csv_file = sys.argv[3]

mri = nib.load(mri_file)
mri_data = mri.get_data()  # mri_data.shape = x, y, number of slices


def extract_csv(input_file):
    regions = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_list = []
            for number in row:
                row_list.append(int(number))
            regions.append(row_list)
    return regions


def perform_lbp_on_regions(input_arr, regions):
    histograms = []
    for roi in regions:
        region = input_arr[roi[1]-EXPAND_PIXELS:roi[3]+EXPAND_PIXELS, roi[2]-EXPAND_PIXELS:roi[4]+EXPAND_PIXELS, roi[0]]

        lbp = feature.local_binary_pattern(region, NUM_POINTS, RADIUS, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NUM_POINTS + 3), range=(0, NUM_POINTS + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram of Local Binary Patterns
        histograms.append(hist)
    return histograms


def flatten_histogram(histograms):
    return [sum(i) / len(histograms) for i in zip(*histograms)]


regions_of_interest = extract_csv(lesion_csv_file)

all_hist = perform_lbp_on_regions(mri_data, regions_of_interest)

flattened_hist = flatten_histogram(all_hist)

with open(output_csv_file, 'wb') as f:
    w = csv.writer(f)
    w.writerow(flattened_hist)
