#!/bin/bash

# Takes in reference (atlas) FLAIR and White Matter files, input data
# folder, and output folder, and registers the WM atlas to the input
# images' image-space.

ref_flair=$1
ref_wm=$2
data_folder=$3
output_folder=$4
for i in $(ls $data_folder);
do
    echo "Registering WM atlas to $i."
    flirt -ref "$data_folder/$i" -in $ref_flair -omat "$output_folder/$i.mat"
    flirt -ref "$data_folder/$i" -in $ref_wm -applyxfm -init "$output_folder/$i.mat" -out "$output_folder/wm_$i"
    echo "Done. Saved as wm_$i.gz"
done
