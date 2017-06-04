Finds lesions on the patient's preprocessed MRI. Currently using [segmentation algorithm from scikit-image](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py).
Preprocess images first to receive a masked image with non-white-matter areas hidden.

TODO: batch+parallel this

Run using 

`python lesion.py [preprocessed, masked image] [output image] [output csv]`

Output CSV is of the format:
`slice_number, bounding_box_left, bb_bottom, bb_right, bb_top`
