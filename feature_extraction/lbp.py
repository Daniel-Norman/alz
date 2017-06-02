import csv
import sys
import nibabel as nib
import numpy as np
import uniform_histogram as uh
import matplotlib.pyplot as plt
from skimage import feature

if len(sys.argv) != 4:
    print 'Expects 2 arguments: registered_wm_atlas_file output_file csv_file'
    quit()

numPoints = 8
radius = 1

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

def calculate_lbp(wm_atlas_data, i, j, k):
    if i == 0 or i == len(wm_atlas_data) - 1 or j == 0 or j == len(wm_atlas_data[0]) - 1:   # out of bounds check
        lbp_data[i][j][k] = 0.0
    else:
        calculated_values = []
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i+1][j][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i+1][j+1][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i][j+1][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i-1][j+1][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i-1][j][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i-1][j-1][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i][j-1][k])
        calculated_values.append(wm_atlas_data[i][j][k] < wm_atlas_data[i+1][j-1][k])
        # print calculated_values
        computed_value = 0
        for power, value in enumerate(calculated_values):
            if value:
                computed_value += 2**power

        lbp_data[i][j][k] = computed_value

        binary = create_binary_string(calculated_values)
        if binary in uh.uniform_histogram:
            uh.uniform_histogram[binary] += 1
        else:
            uh.uniform_histogram["non_uniform"] += 1

def isolate_regions(input_arr, regions_of_interest):
    count = 0
    isolated_slices_array = input_arr
    for i in xrange(len(input_arr)):
        for j in xrange(len(input_arr[0])):
            for k in xrange(len(input_arr[0][0])):
                isolated_slices_array[i][j][k] = 0
    for row in regions_of_interest:
        print row
        for i in xrange(len(input_arr)):
            for j in xrange(len(input_arr[0])):
                for k in xrange(len(input_arr[0][0])):
                    # check if slice number matches
                    # checks if in bounding box in the order left, bottom, right, top
                    # TODO add parameter to expand box
                    if i == row[0] and j >= row[1] and k >= row[2] and j <= row[3] and k <= row[4]:
                        count += 1
                        isolated_slices_array[i][j][k] = input_arr[i][j][k]
    print "count = ", count
    return input_arr

def extract_image(input_arr, regions_of_interest):
    all_images = []
    all_hist = []

    # make it so array is (# slices, 256, 256)
    input_arr = np.swapaxes(input_arr, 0, 2)
    input_arr = np.swapaxes(input_arr, 1, 2)

    isolated_slices_array = isolate_regions(input_arr, regions_of_interest)
    # isolated_slices_array = input_arr

    for i in xrange(len(isolated_slices_array)):
        lbp = feature.local_binary_pattern(isolated_slices_array[i], numPoints, radius, method="uniform")

        (hist, _) = np.histogram(lbp.ravel(), (2 + numPoints * (numPoints - 1)))

        # # normalize the histogram
        # hist = hist.astype("float")
        # hist /= (hist.sum() + 1e-7)

        all_images.append(lbp)
        all_hist.append(hist)
    plt.plot(all_hist[7])
    plt.show()
    # for i in xrange(len(all_images)):
    #     for j in xrange(len(all_images[0])):
    #         for k in xrange(len(all_images[0][0])):
    #             if all_images[i][j][k] > 9:
    #                 print i, j, k
    #                 print all_images[i][j][k]
    return all_images, all_hist

regions_of_interest = extract_csv(sys.argv[3])

all_images, all_hist = extract_image(wm_atlas_data, regions_of_interest)


# for i in xrange(len(wm_atlas_data)):
#     for j in xrange(len(wm_atlas_data[0])):

            # lbp = feature.local_binary_pattern(image, self.numPoints, self.radius, method="uniform")

# Save the masked result, which for some reason needs to be flipped across y axis
# # to match with input MRI
# extracted_mri = nib.Nifti1Image(lbp_data, [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# nib.save(extracted_mri, sys.argv[2])
# print 'Saved as %s' % sys.argv[2]

# for k, v in uh.uniform_histogram.iteritems():
#     print k, v
