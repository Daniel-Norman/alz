Finds lesions on the patient's preprocessed MRI. Currently using [segmentation algorithm from scikit-image](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py).
Make sure to preprocess your images first to receive a masked image with non-white-matter areas hidden.

Run it in batch and parallel by doing this!

`./lesion_batch.sh [# in parallel] [lesion folder] [output folder]`

Run using

`python lesion.py [preprocessed image] [output csv]`

Output CSV is of the format:
`slice_number, bounding_box_left, bb_bottom, bb_right, bb_top`
