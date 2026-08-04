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

# Convert the rendered PPM to PNG, if ImageMagick is available
if [ -f "output.ppm" ] && command -v convert > /dev/null; then
    convert output.ppm output.png
else
    echo "Note: ImageMagick not found or output.ppm missing"
fi