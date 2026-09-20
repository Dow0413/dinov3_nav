#!/usr/bin/env bash
# Download public test images (Unsplash License: free to use, no attribution required).
# Usage: scripts/fetch_test_images.sh [target_dir] [-f to overwrite]
set -euo pipefail

DIR="${1:-/home/dow/DOW/dinov3/test_images}"
FORCE="${2:-}"
mkdir -p "$DIR"

fetch() { # name url
    local dest="$DIR/$1"
    if [[ -s "$dest" && "$FORCE" != "-f" ]]; then
        echo "skip $1 (exists)"
        return
    fi
    curl -sSfL "$2" -o "$dest" && echo "ok   $1  $(stat -c%s "$dest") bytes" \
        || { echo "FAIL $1"; rm -f "$dest"; }
}

fetch street_traffic.jpg "https://images.unsplash.com/photo-1449824913935-59a10b8d2000?w=1280&q=80&fm=jpg"
fetch country_road.jpg   "https://images.unsplash.com/photo-1476234251651-f353703a034d?w=1280&q=80&fm=jpg"
fetch plaza_bean.jpg     "https://images.unsplash.com/photo-1494522855154-9297ac14b55f?w=1280&q=80&fm=jpg"
fetch street_night.jpg   "https://images.unsplash.com/photo-1519501025264-65ba15a82390?w=1280&q=80&fm=jpg"
fetch office_door.jpg     "https://images.unsplash.com/photo-1524758631624-e2822e304c36?w=1280&q=80&fm=jpg"
fetch office_corridor.jpg "https://images.unsplash.com/photo-1497366216548-37526070297c?w=1280&q=80&fm=jpg"

echo "--- file types ---"
file "$DIR"/*.jpg
