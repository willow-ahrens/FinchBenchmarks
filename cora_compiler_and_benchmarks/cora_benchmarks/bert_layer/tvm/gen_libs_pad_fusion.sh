#!/bin/bash

DS=$1
PF=$2
YES="1"
set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

ARGS=" --target cuda --gen-lib --disable-assert --skip-residual"

rm -f ${SCRIPT_DIR}/genlibs/*
if [ $PF == $YES ]; then
    echo "PAD FUSION"
    python3 ${SCRIPT_DIR}/pre_linear.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/qkt.py --dataset $DS $ARGS --sched 1
    python3 ${SCRIPT_DIR}/softmax.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/attn_v.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/post_linear.py --dataset $DS $ARGS
else
    ARGS=$ARGS" --layout-unfused"
    echo "NO PAD FUSION"

    python3 ${SCRIPT_DIR}/pre_linear.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/padding_64to1_add.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/qkt.py --dataset $DS $ARGS --sched 1
    python3 ${SCRIPT_DIR}/padding_32to64.py --dataset $DS $ARGS --padding-mode remove
    python3 ${SCRIPT_DIR}/softmax.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/padding_32to64.py --dataset $DS $ARGS --padding-mode add
    python3 ${SCRIPT_DIR}/attn_v.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/padding_64to1_remove.py --dataset $DS $ARGS
    python3 ${SCRIPT_DIR}/post_linear.py --dataset $DS $ARGS
fi
