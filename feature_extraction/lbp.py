import csv
import sys
import nibabel as nib
import numpy as np
import uniform_histogram as uh
import matplotlib.pyplot as plt
import copy
from skimage import feature

if len(sys.argv) != 5:
    print 'Expects 4 arguments: registered_wm_atlas_file output_file csv_file cognitive_impairment_level'
    quit()

numPoints = 8
radius = 1
expand_pixels = 30

wm_atlas = nib.load(sys.argv[1])
wm_atlas_data = wm_atlas.get_data()
lbp_data = wm_atlas.get_data()
uniform_histogram = [0] * 59 # uniform_histogram[58] is the non uniform bucket
print wm_atlas_data.shape # 256, 256, number of slices

def extract_csv(input_file):
    regions_of_interest = []
    with open(input_file, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in reader:
            row_list = []
            for number in row[0].split(","):
                row_list.append(int(number))
            regions_of_interest.append(row_list)
    return regions_of_interest

def create_binary_string(calculated_values):
    binary = ""
    for value in calculated_values:
        if value == True:
            binary += "1"
        else:
            binary += "0"
    return binary

def calculate_lbp(input_arr, i, j, k, hist):
    if j == 0 or j == len(input_arr[0]) - 1 or k == 0 or k == len(input_arr[0][0]) - 1:   # out of bounds check
        return hist
    else:
        calculated_values = []
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j+1][k])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j+1][k+1])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j][k+1])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j-1][k+1])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j-1][k])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j-1][k-1])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j][k-1])
        calculated_values.append(input_arr[i][j][k] < input_arr[i][j+1][k+1])
        # print calculated_values
        computed_value = 0
        for power, value in enumerate(calculated_values):
            if value:
                computed_value += 2**power

        binary = create_binary_string(calculated_values)
        if binary in uh.uniform_histogram:
            hist[binary] += 1
        else:
            hist["non_uniform"] += 1

        return hist

def isolate_regions(input_arr, regions_of_interest, expand_pixels):
    count = 0
    isolated_slices_array = np.zeros(shape=(len(input_arr), len(input_arr[0]), len(input_arr[0][0])))
    for row in regions_of_interest:
        # print row
        for i in xrange(len(input_arr)):
            for j in xrange(len(input_arr[0])):
                for k in xrange(len(input_arr[0][0])):
                    # check if slice number matches
                    # checks if in bounding box in the order left, bottom, right, top
                    if i == row[0] and j + expand_pixels >= row[1] and k + expand_pixels >= row[2] and j - expand_pixels <= row[3] and k  - expand_pixels <= row[4]:
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

    isolated_slices_array = isolate_regions(input_arr, regions_of_interest, expand_pixels)
    # isolated_slices_array = input_arr # tests without isolating regions

    for i in xrange(len(isolated_slices_array)):
        hist = copy.deepcopy(uh.uniform_histogram)
        for j in xrange(len(isolated_slices_array[0])):
            for k in xrange(len(isolated_slices_array[0][0])):
                hist = calculate_lbp(isolated_slices_array, i, j, k, hist)

        # hist /= hist.sum() # TODO normalize histogram. Prob divide each row by area.
        all_hist.append(hist)
        print hist

    return all_images, all_hist

regions_of_interest = extract_csv(sys.argv[3])

all_images, all_hist = extract_image(wm_atlas_data, regions_of_interest)

keys = all_hist[0].keys()
has_written_cog_level = False

with open(sys.argv[2], 'wb') as f:
    if not has_written_cog_level:
        w = csv.writer(f)
        w.writerow(sys.argv[4])
        has_written_cog_level = True
    w = csv.DictWriter(f, keys)
    w.writeheader()
    w.writerows(all_hist)
