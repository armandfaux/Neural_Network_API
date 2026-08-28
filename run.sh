#!/usr/bin/env bash
set -euo pipefail

echo "Building project..."

mkdir -p bin
javac -d bin src/core/*.java src/data/*.java src/layers/*.java src/tools/*.java

echo "Build successful."
echo "Running project..."

java -cp bin core.Main
