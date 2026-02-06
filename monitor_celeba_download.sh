#!/bin/bash

# Monitor CelebA-HQ dataset download progress

CACHE_DIR="/home/doshlom4/work/final_project/dataset_cache"
CELEBA_DIR="$CACHE_DIR/mattymchen___celeba-hq"

echo "🔍 CelebA-HQ Download Monitor"
echo "================================"
echo "Cache directory: $CACHE_DIR"
echo "CelebA-HQ directory: $CELEBA_DIR"
echo ""

# Function to show sizes
show_sizes() {
    echo "📊 Current Sizes:"
    echo "  Total cache: $(du -sh $CACHE_DIR 2>/dev/null | cut -f1)"
    echo "  CelebA-HQ:   $(du -sh $CELEBA_DIR 2>/dev/null | cut -f1)"
    
    # Show downloads directory if it exists
    if [ -d "$CACHE_DIR/downloads" ]; then
        echo "  Downloads:   $(du -sh $CACHE_DIR/downloads 2>/dev/null | cut -f1)"
    fi
    
    echo ""
    echo "📁 CelebA-HQ Directory Contents:"
    find "$CELEBA_DIR" -type f 2>/dev/null | while read f; do
        echo "  $(ls -lh "$f" | awk '{print $5, $9}')"
    done
    
    echo ""
    echo "🔄 Processes doing I/O:"
    lsof 2>/dev/null | grep "$CELEBA_DIR" | wc -l
    echo "  (Count of open file handles in download directory)"
}

# Show initial status
show_sizes

# If in monitoring mode (argument provided), refresh every 5 seconds
if [ "$1" = "-m" ] || [ "$1" = "--monitor" ]; then
    echo ""
    echo "📡 Monitoring mode (updates every 5 seconds, press Ctrl+C to stop)..."
    echo ""
    while true; do
        sleep 5
        clear
        echo "🔍 CelebA-HQ Download Monitor [$(date '+%H:%M:%S')]"
        echo "================================"
        show_sizes
    done
fi
