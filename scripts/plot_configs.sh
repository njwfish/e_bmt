#!/usr/bin/env bash

IN_DIR=$1
OUT_DIR=$2
REL_METHOD=$3
STYLE_FLAGS=$4

STYLE_OPT=""

if [[ $IN_DIR == "" ]]; then
    echo "Specify in directory"
    exit 1
fi

if [[ $OUT_DIR == "" ]]; then
    echo "Specify out directory"
    exit 1
fi

if [[ $REL_METHOD == "" ]]; then
    REL_METHOD="PUCB_EBH"
fi

if [[ $STYLE_FLAGS != "" ]]; then
    STYLE_OPT="${STYLE_FLAGS}"
fi


EXP_DIR_FLAGS=""

for subdir in $IN_DIR/*/; do
    echo $subdir
    EXP_DIR_FLAGS+="--exp_dir $subdir "
done
CMD="python src/main.py parse-results ${EXP_DIR_FLAGS} --out_dir ${OUT_DIR} --rel_method '${REL_METHOD}' ${STYLE_OPT}"
echo $CMD
eval ${CMD}
