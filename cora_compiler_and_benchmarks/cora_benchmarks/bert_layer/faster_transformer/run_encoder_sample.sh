#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
BIN_DIR=$SCRIPT_DIR/build/bin
set -x
DATA_FILE=$1
BATCH_SIZE=$2
NUM_BATCHES=$3
MAX_SEQ_LEN=$4
NUM_LAYERS=$5
REMOVE_PADDING=$6

rm -f gemm_config.in
$BIN_DIR/encoder_gemm $BATCH_SIZE $MAX_SEQ_LEN 8 64 0 0
$BIN_DIR/encoder_sample $DATA_FILE $NUM_BATCHES $BATCH_SIZE $NUM_LAYERS $MAX_SEQ_LEN 8 64 0 $REMOVE_PADDING 0
