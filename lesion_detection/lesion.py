import sys
import nibabel as nib
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from skimage.filters import threshold_otsu
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
from skimage.feature import blob_doh

if len(sys.argv) != 3:
  print 'Expects 2 arguments: preprocessed_file output_file'
  quit()

mri = nib.load(sys.argv[1])
mri_data = mri.get_data()
mri_shape = mri_data.shape


# Keep only voxels with intensity greater than mean + 2*stddev (ignoring background voxels during calculation)
intensities = []
for i in xrange(mri_shape[0]):
  for j in xrange(mri_shape[1]):
    for k in xrange(mri_shape[2]):
      if mri_data[i][j][k] != 0:
        intensities.append(mri_data[i][j][k])

mean = np.mean(intensities)
std = np.std(intensities)
threshold_lesion = mean + 2*std

for i in xrange(mri_shape[0]):
  for j in xrange(mri_shape[1]):
    for k in xrange(mri_shape[2]):
      if mri_data[i][j][k] < threshold_lesion:
        mri_data[i][j][k] = 0


# Find lesions using image segmentation and labeling, along with blob detection

for k in xrange(mri_shape[2]):
  image = mri_data[:, :, k].copy(order='C')  # Copy is needed for blob detection

  # Perform image segmentation/labeling
  try:
    thresh = threshold_otsu(image)
  except ValueError:
    # Thrown if the image is all one color (i.e. all black background).
    # If so, ignore this slice.
    continue

  bw = closing(image > thresh, square(3))
  cleared = clear_border(bw)
  label_image = label(cleared)

  # Zero out this slice's image. Lesion pixels will be written back as 1.
  mri_data[:, :, k] = 0


  # Perform blob detection to detect lesions that may have a thin "tail"
  # coming off them that otsu thresholding + region.extent thresholding would
  # throw out.

  # TODO: Figure out why blob detection is returning no blobs
  # blobs = blob_doh(image)

  # TODO: Remove plotting part of code after figuring out blob detection

  # fig, ax = plt.subplots(figsize=(10, 6))
  # ax.imshow(image)

  for region in regionprops(label_image):
    minr, minc, maxr, maxc = region.bbox
    width = maxr - minr
    height = maxc - minc
    # TODO: fine tune constants in this, like 20 and 0.3...
    if region.area > 20:
      is_lesion = False
      # Keep only regions that are mostly square (not too thin/tall) and
      # are mostly filled with pixels (using extent, aka pixels/bounding_box_area)
      if width < 2 * height and height < 2 * width and region.extent > 0.3:
        is_lesion = True
      # else:
      #   r,c = region.centroid
      #   blob_radius = 0
      #   for blob in blobs:
      #     y,x,sigma = blob
      #     if np.sqrt((r-y)**2+(c-x)**2) < 20 and sigma > blob_radius:
      #       blob_radius = sigma
      #   if blob_radius > 10:
      #     is_lesion = True
      #     minc = x-blob_radius
      #     maxc = x+blob_radius
      #     minr = y-blob_radius
      #     maxr = y+blob_radius

      if is_lesion:
        # rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        # ax.add_patch(rect)
        for row,col in region.coords:
          mri_data[row, col, k] = 1
        # TODO: export lesion coodinates as CSV for use in determining where to peform LBP
        print 'Lesion at slice %s, bounding box %s' % (k, region.bbox)
  # ax.set_axis_off()
  # plt.tight_layout()
  # plt.show()


print 'Mean: %s\nStd: %s' % (mean, std)
extracted_mri = nib.Nifti1Image(mri_data, [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
nib.save(extracted_mri, sys.argv[2])
print 'Saved as %s' % sys.argv[2]
