#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RUN_NUMBER=$1
MODEL=${2:-convmos}
ARCHITECTURE=${3:-gggl}
VARIABLE=${4:-prec}
TEST_ON_VAL=${5:-False}
N_COMPONENTS=${6:-0.95}
NON_LOCAL_DIST=${7:-5}
N_PARAM_SETS=${8:-20}

WEIGHTED_LOSS=${9:-False}
ALPHA=${10:-0.0}
NAMESPACE="steininger"
# REMO:
MIN_YEAR=2000
MAX_YEAR=2015
MAX_TRAIN_YEAR=2009
MAX_VAL_YEAR=2010

### Check if this script has acceptable parameters

if [ -z "$RUN_NUMBER" ] && [ -z "$MODEL" ] && [ -z "$ARCHITECTURE" ] && [ -z "$VARIABLE" ] && [ -z "$WEIGHTED_LOSS" ] && [ -z "$ALPHA" ]
then
    echo "Please specify all parameters (RUN_NUMBER, MODEL, ARCHITECTURE, VARIABLE, WEIGHTED_LOSS, ALPHA)"
    echo "Aborting..."
    exit 1
fi

### Build some variables based on the parameters

ALPHA_STR=$(echo $ALPHA | sed -e "s/\./-/")

if [ "$TEST_ON_VAL" = "True" ]
then
    NAME="hpsearch-"
    ### NOTE: THIS HERE SPECIFIES THE VALIDATION DATASET!!!
    MAX_YEAR=2010
    MAX_TRAIN_YEAR=2008
    MAX_VAL_YEAR=2009
else
    NAME=""
fi

if [ "$WEIGHTED_LOSS" = "True" ]
then
    NAME="${NAME}${MODEL}-$VARIABLE-wl-$ALPHA_STR-$RUN_NUMBER"
else
    NAME="$NAME$MODEL-$VARIABLE-$RUN_NUMBER"
fi

JOB_NAME="sdnext-$NAME"
echo "Job: sdnext-$NAME"

### Check whether the config for these parameters already exist

CONFIG_DIR="$DIR/../configs/"
CONFIG_NAME="config-$NAME.ini"
CONFIG_PATH="$CONFIG_DIR$CONFIG_NAME"

if [ -f "$CONFIG_PATH" ]
then
    echo "Config file $CONFIG_PATH already exists"
    echo "Aborting..."
    exit 1
fi

### Delete old job, if there is already one

job_exists=$(kubectl get job -n $NAMESPACE | grep $JOB_NAME)

if [ -n "$job_exists" ]
then
    echo "job already exists"
    echo "Delete old job? [y/n]"
    read delete
    if [ $delete = "y" ]
    then
        kubectl delete job -n $NAMESPACE $JOB_NAME
        echo "Deleted job $JOB_NAME"
    else
        echo "Aborting..."
        exit 1
    fi
fi

### Check if model is a baseline
MODEL_IS_BASELINE='False'
if [ "$MODEL" = "linear" ] || [ "$MODEL" = "linear-nonlocal" ] || [ "$MODEL" = "rf" ] || [ "$MODEL" = "rf-nonlocal" ] || [ "$MODEL" = "svm" ] || [ "$MODEL" = "svm-nonlocal" ]
then
    MODEL_IS_BASELINE='True'
fi

### Create config

if [ "$VARIABLE" = "temp" ]
then
    CONFIG_TEMPLATE="$DIR/config_template_remo_temp.ini"
elif [ "$VARIABLE" = "prec" ]
then
    if [ "$MODEL_IS_BASELINE" = "True" ]
    then
        CONFIG_TEMPLATE="$DIR/config_template_remo_prec_baseline.ini"
    else
        CONFIG_TEMPLATE="$DIR/config_template_remo_prec.ini"
    fi
else
    echo "Unknown Variable $VARIABLE (available: temp, prec)"
    exit 1
fi

start=$(cat $CONFIG_TEMPLATE | sed -e "s/\${NAME}/$NAME/" -e "s/\${MODEL}/$MODEL/" -e "s/\${ARCHITECTURE}/$ARCHITECTURE/" -e "s/\${WEIGHTED_LOSS}/$WEIGHTED_LOSS/" -e "s/\${ALPHA}/$ALPHA/" -e "s/\${MIN_YEAR}/$MIN_YEAR/" -e "s/\${MAX_YEAR}/$MAX_YEAR/" -e "s/\${MAX_TRAIN_YEAR}/$MAX_TRAIN_YEAR/" -e "s/\${MAX_VAL_YEAR}/$MAX_VAL_YEAR/" -e "s/\${N_COMPONENTS}/$N_COMPONENTS/" -e "s/\${NON_LOCAL_DIST}/$NON_LOCAL_DIST/" -e "s/\${N_PARAM_SETS}/$N_PARAM_SETS/" > $CONFIG_PATH)

### Create ConfigMap in Kubernetes
CONFIGMAP_NAME=$(echo "sdnext-$CONFIG_NAME" | sed -e "s/\.ini//")
kubectl create configmap -n $NAMESPACE $CONFIGMAP_NAME --from-file=$CONFIG_PATH

NON_LOCAL_FLAG=''
if [ "$MODEL_IS_BASELINE" = "True" ]
then
    K8S_TEMPLATE="$DIR/k8s_baseline_job_template.yml"
else
    K8S_TEMPLATE="$DIR/k8s_job_template.yml"
fi

start=$(cat $K8S_TEMPLATE | sed -e "s/\${JOB_NAME}/$JOB_NAME/" -e "s/\${NAMESPACE}/$NAMESPACE/" -e "s/\${CONFIGMAP_NAME}/$CONFIGMAP_NAME/" -e "s/\${CONFIG_NAME}/$CONFIG_NAME/" -e "s/\${NON_LOCAL_FLAG}/$NON_LOCAL_FLAG/" | kubectl apply -f -)
echo $start
echo "Started job $JOB_NAME"
