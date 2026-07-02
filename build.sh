#!/bin/bash
set -e
g++ -O2 -std=c++17 -I include src/main.cpp -o vgg_sim
echo "Built ./vgg_sim"
echo ""
echo "Usage:"
echo "  ./vgg_sim <export_dir> <array_rows> <array_cols> <num_images>"
echo ""
echo "Example (real Colab export, 16x16 array, 50 images):"
echo "  ./vgg_sim export 16 16 50"
echo ""
echo "Example (dummy smoke test data, already generated):"
echo "  ./vgg_sim test_export 16 16 3"
