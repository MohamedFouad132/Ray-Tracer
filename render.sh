#!/bin/bash
# render.sh
# Convenience wrapper for running the ray tracer without manually building first.
# Builds the project only if it hasn't been built yet, then forwards all arguments
# directly to the raytracer executable.
# Usage: ./render.sh [options]
# Run ./render.sh --help to see available options.

cd "$(dirname "$0")"

if [ ! -f "./build/raytracer" ]; then
    echo "Building raytracer..."
    mkdir -p build
    cd build
    cmake .. > /dev/null
    make > /dev/null
    cd ..
fi

./build/raytracer "$@"

# Convert the rendered PPM to PNG and open it, if a display is available
if [ -f "output.ppm" ] && command -v xdg-open > /dev/null && [ -n "$DISPLAY" ]; then
    if command -v convert > /dev/null; then
        convert output.ppm output.png
        xdg-open output.png
    else
        echo "Note: ImageMagick not found, skipping PNG conversion. Install with: sudo apt-get install imagemagick"
        xdg-open output.ppm
    fi
fi