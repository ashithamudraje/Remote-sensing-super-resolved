#!/bin/bash

# Define the list of directories to be removed
directories=(
    "/ds/images/BigEarthNet/BigEarthNet-S2/BigEarthNet_rgb_split/super_resolved_images/Seesr_all_images"
    # Add more directories as needed
)

# Loop through each directory and remove it if it exists
for dir in "${directories[@]}"; do
    if [ -d "$dir" ]; then
        rm -rf "$dir"
        echo "Removed directory $dir"
    else
        echo "Directory $dir does not exist"
    fi
done

echo "Script execution completed."