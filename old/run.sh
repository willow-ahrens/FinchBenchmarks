#!/bin/bash

bash -e spmspv.sh
bash -e triangle.sh
bash -e conv.sh
bash -e alpha.sh
bash -e all_pairs.sh

bash -e spmspv_plot.sh
bash -e triangle_plot.sh
bash -e conv_plot.sh
bash -e alpha_plot.sh
bash -e all_pairs_plot.sh