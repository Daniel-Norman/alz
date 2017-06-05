import csv
import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import copy
from skimage import feature

NUM_POINTS = 8
RADIUS = 1
EXPAND_PIXELS = 10

if len(sys.argv) != 4:
    print 'Expects 3 arguments: mri_file lesion_csv output_csv'
    quit()

mri_file = sys.argv[1]
lesion_csv_file = sys.argv[2]
output_csv_file = sys.argv[3]

mri = nib.load(mri_file)
mri_data = mri.get_data()
print mri_data.shape # x, y, number of slices

def extract_csv(input_file):
    regions_of_interest = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            row_list = []
            for number in row:
                row_list.append(int(number))
            regions_of_interest.append(row_list)
    return regions_of_interest

def isolate_regions(input_arr, regions_of_interest, EXPAND_PIXELS):
    count = 0
    isolated_slices_array = np.zeros(shape=(len(input_arr), len(input_arr[0]), len(input_arr[0][0])))
    for row in regions_of_interest:
        # print row
        for i in xrange(len(input_arr)):
            for j in xrange(len(input_arr[0])):
                for k in xrange(len(input_arr[0][0])):
                    # check if slice number matches
                    # checks if in bounding box in the order left, bottom, right, top
                    if i == row[0] and j + EXPAND_PIXELS >= row[1] and k + EXPAND_PIXELS >= row[2] and j - EXPAND_PIXELS <= row[3] and k  - EXPAND_PIXELS <= row[4]:
                        count += 1
                        isolated_slices_array[i][j][k] = input_arr[i][j][k]
    print "Total area of bounding boxes =", count
    return input_arr

def extract_image(input_arr, regions_of_interest):
    all_images = []
    all_hist = []

    # make it so array is (# slices, 256, 256)
    input_arr = np.swapaxes(input_arr, 0, 2)
    input_arr = np.swapaxes(input_arr, 1, 2)

    isolated_slices_array = isolate_regions(input_arr, regions_of_interest, EXPAND_PIXELS)

    for i in xrange(len(isolated_slices_array)):
        lbp = feature.local_binary_pattern(isolated_slices_array[i], NUM_POINTS, RADIUS, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, NUM_POINTS + 3), range=(0, NUM_POINTS + 2))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= hist.sum()

        # return the histogram of Local Binary Patterns
        all_hist.append(hist)
        print hist

    return all_images, all_hist

def flatten_histogram(all_hist):
    return [sum(i) / len(all_hist) for i in zip(*all_hist)]

regions_of_interest = extract_csv(lesion_csv_file)

all_images, all_hist = extract_image(mri_data, regions_of_interest)

flattened_hist = flatten_histogram(all_hist)

print flattened_hist
print len(all_hist)

with open(output_csv_file, 'wb') as f:
    w = csv.writer(f)
    w.writerow(flattened_hist)
