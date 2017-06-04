#!/bin/bash

# Takes in amount to run in parallel, reference (atlas) FLAIR and White Matter files,
# input data folder, and output folder, and registers the WM atlas to the input
# images' image-space.

parallel=$1
ref_flair=$2
ref_wm=$3
data_folder=$4
output_folder=$5
index=0
for i in $(ls $data_folder);
do
    echo "Registering WM atlas to $i."
    (flirt -ref "$data_folder/$i" -in $ref_flair -omat "$output_folder/$i.mat" && flirt -ref "$data_folder/$i" -in $ref_wm -applyxfm -init "$output_folder/$i.mat" -out "$output_folder/wm_$i" && echo "Done, saved wm_$i.gz") &
    ((index++))
    if [ $index = $parallel ]; then
        index=0
        wait;
    fi
done
wait
