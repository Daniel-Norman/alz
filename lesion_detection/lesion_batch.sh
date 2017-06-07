#!/bin/bash

parallel=$1
masked_input_folder=$2
output_folder=$3
index=0
for i in $(ls $masked_input_folder);
do
    echo "Finding lesions in $i..."
    (python lesion.py "$masked_input_folder/$i" "$output_folder/lesions_$i.csv") &
    ((index++))
    if [ $index = $parallel ]; then
        index=0
        wait
    fi
done
wait
