Performs preprocessing on the patient's MRI. Requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) installed.

First register using

`wm_register.sh [FLAIR atlas file] [WM atlas file] [input data folder] [output folder]`

to receive versions of the white matter atlas file, each registered to an input MRI's image space.


Then use the registered WM files to mask each input MRI using

`python wm_mask.py [threshold, between 0 to 1] [registered wm file] [input MRI file] [output file]`

This output should include only white matter regions, stripping away non-WM (including the skull).
These can now be run through lesion detection.
