#!/bin/bash

# Set the directory containing the shell scripts
directory="/path/to/directory"

# Change to the directory
cd "$directory"

# Iterate over all shell scripts in the directory
for script in *.sh; do
    # Check if the file is actually a shell script and filename starts with "test" or "train"
    if [[ -f $script && -x $script && ($script == test* || $script == train*) ]]; then
        # Run the shell script
        "./$script"
    fi
done