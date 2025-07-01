#!/bin/bash

# Directory containing the parquet files
INPUT_DIR="../data/processed_$(date +%y%m%d)"

# Process all 5Hz parquet files
for file in "$INPUT_DIR"/*_5Hz.parquet; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        python summary_metrics.py \
            --input "$file" \
            --verbose
    fi
done

echo "All files processed!"