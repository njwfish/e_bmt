#!/usr/bin/env bash

DIR=$1
PROC=$2
OUT_DIR=$3

if [[ $DIR == "" ]]; then
    echo "Specify config directory"
    exit 1
fi

if [[ $PROC == "" ]]; then
    echo "Specify number of processes"
    exit 1
fi

if [[ $OUT_DIR == "" ]]; then
    echo "Specify output directory"
    exit 1
fi

function get_name() {
    local method=$(basename -- "$1")
    method=${method%.json}
    echo "$method"
}

for config in $DIR/*.json; do
    method=$(get_name $config)
    echo "Testing ${method}"
    python -W ignore src/main.py exp $config $OUT_DIR/$method --processes $PROC
done
