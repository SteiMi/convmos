#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

NUM_RUNS=20

for i in `seq 1 $NUM_RUNS`;
do
    echo "Starting run $i"
    $DIR/start_k8s_job.sh nlpcr-$i linear-nonlocal
done
