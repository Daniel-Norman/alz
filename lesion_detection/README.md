Finds lesions on the patient's preprocessed MRI. Currently using [segmentation algorithm from scikit-image](http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py).

Run using 

`python lesion.py [preprocessed image] [output image] [output csv]`

Output CSV is of the format:
`slice_number, bounding_box_left, bb_bottom, bb_right, bb_top`
