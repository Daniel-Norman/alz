import sys
import nibabel as nib
import numpy as np

if len(sys.argv) != 3:
    print 'Expects 2 arguments: registered_wm_atlas_file output_file'
    quit()

wm_atlas = nib.load(sys.argv[1])
wm_atlas_data = wm_atlas.get_data()
lbp_data = wm_atlas.get_data()
print wm_atlas_data.shape # 256, 256, number of slices

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
        computed_value = 0
        for power, value in enumerate(calculated_values):
            if value:
                computed_value += 2**power
        lbp_data[i][j][k] = computed_value

for i in xrange(len(wm_atlas_data)):
    for j in xrange(len(wm_atlas_data[0])):
        for k in xrange(len(wm_atlas_data[0][0])):
            calculate_lbp(wm_atlas_data, i, j, k)

# Save the masked result, which for some reason needs to be flipped across y axis
# to match with input MRI
extracted_mri = nib.Nifti1Image(lbp_data, [[-1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
nib.save(extracted_mri, sys.argv[2])
print 'Saved as %s' % sys.argv[2]
