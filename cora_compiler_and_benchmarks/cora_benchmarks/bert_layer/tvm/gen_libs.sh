#!/bin/bash

DS=$1
BP=$2
MS=$3
OP=$4
YES="1"
set -x
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

rm -f $SCRIPT_DIR/genlibs/*

EXTRA_ARGS="--disable-assert "
if [ $OP == $YES ]; then
    EXTRA_ARGS="--only-prep-code "
fi

if [ $MS == $YES ]; then
    EXTRA_ARGS=$EXTRA_ARGS"--skip-residual"
fi

if [ $MS == $YES ]; then
    echo -ne ""
else
    python3 ${SCRIPT_DIR}/norm_add.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    python3 ${SCRIPT_DIR}/pre_linear.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    python3 ${SCRIPT_DIR}/memset.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    python3 ${SCRIPT_DIR}/post_linear.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 1
    python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 2
    python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 3
    python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 4
    python3 ${SCRIPT_DIR}/ff2.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 5
    python3 ${SCRIPT_DIR}/ff1.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 1
    python3 ${SCRIPT_DIR}/ff1.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 2
fi

if [ $MS == $YES ]; then
    if [ $BP == $YES ]; then
	echo "Masked bin packed operators not implemented"
	exit 1
    else
	echo "1.2"
	python3 ${SCRIPT_DIR}/masked_qkt.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/masked_attn_v.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/masked_softmax.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    fi
else
    echo "2"
    python3 ${SCRIPT_DIR}/softmax.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    if [ $BP == $YES ]; then
	echo "2.1"
	python3 ${SCRIPT_DIR}/qkt_bin_packed.py --hfuse --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
	python3 ${SCRIPT_DIR}/attn_v_bin_packed.py --hfuse --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    else
	echo "2.2"
	python3 ${SCRIPT_DIR}/qkt.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 1
	python3 ${SCRIPT_DIR}/qkt.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS --sched 2
	python3 ${SCRIPT_DIR}/attn_v.py --target cuda --dataset $DS --gen-lib $EXTRA_ARGS
    fi
fi
