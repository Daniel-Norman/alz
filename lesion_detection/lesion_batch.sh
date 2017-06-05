#!/bin/bash

parallel=$1
lesion_folder=$2
output_folder=$3
index=0
for i in $(ls $lesion_folder);
do
    echo "Finding lesions in $i..."
    (python lesion.py "$lesion_folder/$i" "$output_folder/lesion$i.csv" && echo "Done. Saved as $output_folder/lesion$i.csv") &
    ((index++))
    if [ $index = $parallel ]; then
        index=0
        wait
    fi
done
wait
