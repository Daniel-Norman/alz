#!/bin/bash

parallel=$1
mri_folder=$2
lesion_csv_folder=$3
output_folder=$4
index=0
for i in $(ls $mri_folder);
do
    echo "Applying LBP to $i..."
    number=$(echo "$i" | tr -dc '0-9')
    (python lbp.py "$mri_folder/$i" "$lesion_csv_folder/lesions_masked_$number.nii.csv" "$output_folder/lbp_histogram_$number.csv" && echo "Done. Saved as $output_folder/lbp_histogram_$number.csv") &
    ((index++))
    if [ $index = $parallel ]; then
        index=0
        wait
    fi
done
wait
