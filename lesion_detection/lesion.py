import sys
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import csv

from scipy.ndimage.filters import gaussian_filter1d
from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square

# TODO fine tune constants
# sigma used in blurred image
BLUR_STRENGTH = 1
# max ratio of either height/width or width/height for a region to be a lesion
LESION_HW_RATIO = 2
# minimum percent of a region's filled pixels / bounding box area to be a lesion
LESION_MIN_EXTENT = 0.3
# Ratio of lesion area to total slice area, to filter out small areas that are likely not a lesion
LESION_MIN_AREA_RATIO = 20.0 / (256*256)
# Maximum distance between the centroids of some region in original and region in blur, to group them as the same region
BLUR_SAME_REGION_DISTANCE_RATIO = 7.0 / 256

if len(sys.argv) != 4:
    print 'Expects 2 arguments: preprocessed_image output_image output_csv'
    quit()

should_plot = False

mri = nib.load(sys.argv[1])
mri_data = mri.get_data()
slice_length = mri_data.shape[0]
slice_area = mri_data.shape[0]*mri.shape[1]

# Perform a horizontal blur of each slice, to get rid of thin vertical lines
mri_data_blur = gaussian_filter1d(mri_data, sigma=BLUR_STRENGTH, axis=0)


# Keep only voxels with intensity greater than mean + 2*stddev (ignoring background voxels during calculation)
def threshold(scan):
    intensities = []
    for i in xrange(scan.shape[0]):
        for j in xrange(scan.shape[1]):
            for k in xrange(scan.shape[2]):
                if scan[i][j][k] != 0:
                    intensities.append(scan[i][j][k])
    mean = np.mean(intensities)
    std = np.std(intensities)
    threshold_lesion = mean + 2 * std
    for i in xrange(scan.shape[0]):
        for j in xrange(scan.shape[1]):
            for k in xrange(scan.shape[2]):
                if scan[i][j][k] < threshold_lesion:
                    scan[i][j][k] = 0


def could_region_be_lesion(reg):
    min_r, min_c, max_r, max_c = reg.bbox
    w = max_r - min_r
    h = max_c - min_c
    return w < LESION_HW_RATIO * h and h < LESION_HW_RATIO * w and reg.extent > LESION_MIN_EXTENT

# Threshold original image
threshold(mri_data)
threshold(mri_data_blur)

lesions = []

# Find lesions using image segmentation and labeling
for k in xrange(mri_data.shape[2]):
    image = mri_data[:, :, k]
    image_blur = mri_data_blur[:, :, k]

    # TODO: Remove plotting part of code after done fine tuning
    if should_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_axis_off()
        plt.tight_layout()
        ax.imshow(image)

    # Perform image segmentation/labeling
    try:
        bw_image = closing(image > threshold_otsu(image), square(3))
        label_image = label(clear_border(bw_image))

        bw_image_blur = closing(image_blur > threshold_otsu(image_blur), square(3))
        label_image_blur = label(clear_border(bw_image_blur))
    except ValueError:
        # Thrown if the image is all one color (i.e. all black background).
        # If so, ignore this slice.
        continue

    # Zero out this slice's image. Lesion pixels will be written back as 1.
    mri_data[:, :, k] = 0
    mri_data_blur[:, :, k] = 0

    for region in regionprops(label_image):
        bbox = region.bbox
        minr, minc, maxr, maxc = bbox
        centroid = region.centroid
        width = maxr - minr
        height = maxc - minc
        is_lesion = 0
        if region.area > (LESION_MIN_AREA_RATIO*slice_area):
            if could_region_be_lesion(region):
                is_lesion = 1
            else:
                # If the region in original image was thought to not be a lesion, double check with the blurred
                # version. The blurred image's lesions may like this specific region, because blurring gets
                # rid of thin vertical lines that would otherwise make the region's "extent" too low.
                for region_blur in regionprops(label_image_blur):
                    centroid_blur = region_blur.centroid
                    centroid_distance = np.sqrt((centroid[0]-centroid_blur[0])**2+(centroid[1]-centroid_blur[1])**2)
                    if region_blur.area > (LESION_MIN_AREA_RATIO*slice_area)\
                            and could_region_be_lesion(region_blur)\
                            and centroid_distance < (BLUR_SAME_REGION_DISTANCE_RATIO*slice_length):
                        region = region_blur
                        bbox = region_blur.bbox
                        is_lesion = 2
                        break

            if is_lesion != 0:
                # TODO: remove plotting
                if should_plot:
                    minr, minc, maxr, maxc = bbox
                    rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                              fill=False,
                                              edgecolor=('red' if is_lesion == 1 else 'green'),
                                              linewidth=2)
                    ax.add_patch(rect)

                for row, col in region.coords:
                    mri_data[row, col, k] = 1
                lesions.append([k, int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])])

    if should_plot:
        plt.show()

with open(sys.argv[3], 'wb') as lesion_csv:
    csv_writer = csv.writer(lesion_csv)
    for lesion in lesions:
        csv_writer.writerow(lesion)

extracted_mri = nib.Nifti1Image(mri_data, [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
nib.save(extracted_mri, sys.argv[2])
print 'Saved image as %s and CSV as %s.' % (sys.argv[2], sys.argv[3])
