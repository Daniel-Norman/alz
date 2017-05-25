import sys
import nibabel as nib
import numpy as np

if len(sys.argv) != 5:
  print 'Expects 4 arguments: threshold registered_wm_atlas_file mri_file output_file'
  quit()

threshold = float(sys.argv[1])

wm_atlas = nib.load(sys.argv[2])
wm_atlas_data = wm_atlas.get_data()
mri = nib.load(sys.argv[3])
mri_data = mri.get_data()

print 'Using %s with threshold > %s to mask %s' % (sys.argv[2], sys.argv[1], sys.argv[3])
for i in xrange(len(mri_data)):
  for j in xrange(len(mri_data[0])):
    for k in xrange(len(mri_data[0][0])):
      if wm_atlas_data[i][j][k] < threshold:
        mri_data[i][j][k] = 0

# Save the masked result, which for some reason needs to be flipped across y axis
# to match with input MRI
extracted_mri = nib.Nifti1Image(mri_data, [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
nib.save(extracted_mri, sys.argv[4])
print 'Saved as %s' % sys.argv[4]