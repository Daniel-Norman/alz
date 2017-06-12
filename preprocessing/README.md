Performs preprocessing on the patient's MRI. Requires [FSL](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FSL) installed.
Works in parallel on a batch of files.
We used the 1.0mm FLAIR and WM atlas files from [Brainder](https://brainder.org/download/flair/).
Please rename all patient .nii files to a single number per patient followed by .nii, such as `1234.nii`.

*Note: Please copy all scripts to the directory containing your data folders before running.*

First register using

`./wm_batch_register.sh [# in parallel] [FLAIR atlas file] [WM atlas file] [MRI scan folder] [wm regsiter output folder]`

to receive versions of the white matter atlas file, each registered to an input MRI's image space.


Then use the registered WM files to mask each input MRI using

`./wm_batch_mask.sh [# in parallel] [threshold, 0 to 1.0] [wm register output folder] [MRI scan folder] [mask output folder]`


While the paper suggests threshold 0.5, this seemed to hide a lot of lesions. We recommend threshold 0.3.
For full sequential execution, use 1 for the # in parallel.

This output should include only white matter regions, stripping away non-WM (including the skull).
These can now be run through lesion detection.
