#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# NUM_RUNS="20"
NUM_RUNS="20"
# Try all permutations of local networks and global networks of lengths up to 4
ARCHITECTURES=( {l,g} {l,g}{l,g} {l,g}{l,g}{l,g} {l,g}{l,g}{l,g}{l,g} )

for a in "${ARCHITECTURES[@]}";
do
    for i in `seq 1 $NUM_RUNS`;
    do
        echo "Starting run $a-$i"
        $DIR/start_k8s_job.sh $a-$i convmos $a prec True
        # $DIR/start_k8s_job.sh ablationmore64-$a-$i -a $a -z azure -g UNet -t -s
    done
done
