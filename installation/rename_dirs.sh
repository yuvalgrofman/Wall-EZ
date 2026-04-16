#!/bin/bash

# Target directory (defaults to current directory if no argument is passed)
TARGET_DIR="${1:-.}"

# Verify the directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo "Error: Directory '$TARGET_DIR' not found."
    exit 1
fi

cd "$TARGET_DIR" || exit

# Loop through all directories that consist entirely of numbers
for dir in [0-9]*/; do
    # Skip if the glob didn't match anything (e.g., no numeric directories exist)
    if [ ! -d "$dir" ]; then
        continue
    fi

    # Strip the trailing slash from the directory name to get the raw timestamp
    timestamp="${dir%/}"

    # Convert the timestamp to Israel time.
    # TZ="Asia/Jerusalem" forces the timezone to Israel Standard/Daylight Time.
    # Note: This uses GNU 'date' (standard on Linux).
    idt_name=$(TZ="Asia/Jerusalem" date -d "@$timestamp" +"%Y-%m-%d_%H-%M-%S" 2>/dev/null)

    # Check if the date conversion was successful before renaming
    if [ -n "$idt_name" ]; then
        echo "Renaming: $timestamp  ->  $idt_name"
        mv "$timestamp" "$idt_name"
    else
        echo "Warning: Could not parse '$timestamp' as a valid date. Skipping."
    fi
done

echo "Done!"