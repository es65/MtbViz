#!/bin/bash

# Directory containing the parquet files
INPUT_DIR="../data/interim"
OUTPUT_DIR="../data/processed_250530"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Process all 5Hz parquet files
for file in "$INPUT_DIR"/*_5Hz.parquet; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        python build_features.py \
            --input "$file" \
            --output-dir "$OUTPUT_DIR" \
            --verbose
    fi
done

echo "All files processed!"