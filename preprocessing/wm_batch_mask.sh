#!/bin/bash

parallel=$1
threshold=$2
wm_folder=$3
mri_folder=$4
output_folder=$5
index=0
for i in $(ls $mri_folder);
do
    echo "Masking $i..."
    (python wm_mask.py $threshold "$wm_folder/wm_$i.gz" "$mri_folder/$i" "$output_folder/masked_$i" && echo "Done. Saved as masked_$i.gz") &
    ((index++))
    if [ $index = $parallel ]; then
        index=0
        wait
    fi
done
wait
