#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# NUM_RUNS=20
# NUM_RUNS=10
NUM_RUNS=5

for i in `seq 1 $NUM_RUNS`;
do
    echo "Starting run $i"
    # $DIR/start_k8s_job.sh UNet-$i -m UNet
    # $DIR/start_k8s_job.sh unet-gl-$i -a gl -g UNet -o none 
    # $DIR/start_k8s_job.sh resnet101-$i -m ResNet101
    # $DIR/start_k8s_job.sh resnet101-gl-$i -a gl -g ResNet101
    # ees = extreme early stopping = early stopping based on performance on extreme values
    # $DIR/start_k8s_job.sh ees2-$i -z azure -x 1.0 -w -s && sleep 3
    # $DIR/start_k8s_job.sh dlsearch-ees2-$i -z azure -x 0.7 -w -t -s && sleep 3
    # $DIR/start_k8s_job.sh dlsearch-nes-$i -z azure -x 0.0 -e 400 -w -t && sleep 3
    # $DIR/start_k8s_job.sh nes-$i -z azure -x 1.0 -e 400 -w && sleep 3
    # $DIR/start_k8s_job.sh nes-$i -z azure -e 400 && sleep 3
    # $DIR/start_k8s_job.sh timetest-$i -m ResNet101 -s
    # $DIR/start_k8s_job.sh timetest-$i -m rf-nonlocal
    # $DIR/start_k8s_job.sh timetest-$i -m linear-nonlocal
    $DIR/start_k8s_job.sh timetest-$i -m linear
    # az container delete --resource-group EU-N -y --name sdnext-convmos-prec-wl-1-0-ees2-$i
done
