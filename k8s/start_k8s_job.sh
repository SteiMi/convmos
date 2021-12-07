#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

RUN_NUMBER=$1
# MODEL=${2:-convmos}
# ARCHITECTURE=${3:-gggl}
# LOCAL_MODULE=${4:-LocalNet}
# GLOBAL_MODULE=${5:-GlobalNet}
# OUTPUT_ACTIVATION=${6:-ReLU}
# VARIABLE=${4:-prec}
# TEST_ON_VAL=${5:-False}
# N_COMPONENTS=${6:-0.95}
# NON_LOCAL_DIST=${7:-5}
# N_PARAM_SETS=${8:-20}
# WEIGHTED_LOSS=${9:-False}
# ALPHA=${10:-0.0}

MODEL="convmos"
ARCHITECTURE="gggl"
LOCAL_MODULE="LocalNet"
GLOBAL_MODULE="GlobalNet"
OUTPUT_ACTIVATION="ReLU"
VARIABLE="prec"
TEST_ON_VAL="False"
N_COMPONENTS="0.95"
NON_LOCAL_DIST="5"
N_PARAM_SETS="20"
WEIGHTED_LOSS="False"
ALPHA="0.0"
WHERE="kubernetes"
EPOCHS="100000"
EARLY_STOPPING="False"

NAMESPACE="steininger"
# REMO:
MIN_YEAR=2000
MAX_YEAR=2015
MAX_TRAIN_YEAR=2009
MAX_VAL_YEAR=2010

display_usage() { 
	echo -e "\nUsage:\n./start_k8s_job.sh run_number [-r] [-m] \n"
    echo -e "Schedules a run called run_number in Kubernetes"
    echo -e "-m: Model to use"
    echo -e "-a: Architecture to use (for model ConvMOS)"
    echo -e "-l: Local module to use (for model ConvMOS)"
    echo -e "-g: Global module to use (for model ConvMOS)"
    echo -e "-o: Output activation to use (for model ConvMOS)"
    echo -e "-v: Variable to improve"
    echo -e "-t: Test on validation set"
    echo -e "-c: N_Components to use (Baselines only)"
    echo -e "-d: Non-local distance (Baselines only)"
    echo -e "-p: Number of parameter sets to try (Baselines only)"
    echo -e "-w: Activate weighted loss"
    echo -e "-x: Alpha parameter for weighted loss"
    echo -e "-z: Where it should be run (kubernetes or azure)"
    echo -e "-e: Maximum number of epochs to train"
    echo -e "-s: Stop early based on validation metrics"
}

display_opts() {
    echo -e "RUN_NUMBER =\t\t$RUN_NUMBER"
    echo -e "MODEL =\t\t\t$MODEL"
    echo -e "ARCHITECTURE =\t\t$ARCHITECTURE"
    echo -e "LOCAL_MODULE =\t\t$LOCAL_MODULE"
    echo -e "GLOBAL_MODULE =\t\t$GLOBAL_MODULE"
    echo -e "OUTPUT_ACTIVATION =\t$OUTPUT_ACTIVATION"
    echo -e "VARIABLE =\t\t$VARIABLE"
    echo -e "TEST_ON_VAL =\t\t$TEST_ON_VAL"
    echo -e "N_COMPONENTS =\t\t$N_COMPONENTS"
    echo -e "NON_LOCAL_DIST =\t$NON_LOCAL_DIST"
    echo -e "N_PARAM_SETS =\t\t$N_PARAM_SETS"
    echo -e "WEIGHTED_LOSS =\t\t$WEIGHTED_LOSS"
    echo -e "ALPHA =\t\t\t$ALPHA"
    echo -e "WHERE =\t\t\t$WHERE"
    echo -e "EPOCHS =\t\t$EPOCHS"
    echo -e "EARLY_STOPPING =\t$EARLY_STOPPING"
}

while getopts 'm:a:l:g:o:v:tc:d:p:wx:z:e:s' flag ${@:2}; do
  case "${flag}" in
    m) MODEL=${OPTARG};;
    a) ARCHITECTURE=${OPTARG};;
    l) LOCAL_MODULE=${OPTARG};;
    g) GLOBAL_MODULE=${OPTARG};;
    o) OUTPUT_ACTIVATION=${OPTARG};;
    v) VARIABLE=${OPTARG};;
    t) TEST_ON_VAL="True";;
    c) N_COMPONENTS=${OPTARG};;
    d) NON_LOCAL_DIST=${OPTARG};;
    p) N_PARAM_SETS=${OPTARG};;
    w) WEIGHTED_LOSS="True";;
    x) ALPHA=${OPTARG};;
    z) WHERE=${OPTARG};;
    e) EPOCHS=${OPTARG};;
    s) EARLY_STOPPING="True";;
    *) display_usage
       exit 1 ;;
  esac
done

display_opts

### Check if this script has acceptable parameters

if [ -z "$RUN_NUMBER" ]
then
    echo "Please specify a RUN_NUMBER!"
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

NAME=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')

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

if [ -n "$job_exists" ] && [ "$WHERE" = "kubernetes" ]
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
    # E-OBS rr precipitation
    if [ "$MODEL_IS_BASELINE" = "True" ]
    then
        CONFIG_TEMPLATE="$DIR/config_template_remo_prec_baseline.ini"
    else
        CONFIG_TEMPLATE="$DIR/config_template_remo_prec.ini"
    fi
elif [ "$VARIABLE" = "prec-era5-cp" ]
then
    # ERA5 convective precipitation
    CONFIG_TEMPLATE="$DIR/config_template_remo_prec_era5_cp.ini"
elif [ "$VARIABLE" = "prec-era5-lsp" ]
then
    # ERA5 large scale precipitation
    CONFIG_TEMPLATE="$DIR/config_template_remo_prec_era5_lsp.ini"
elif [ "$VARIABLE" = "prec-era5-sf" ]
then
    # ERA5 snowfall
    CONFIG_TEMPLATE="$DIR/config_template_remo_prec_era5_sf.ini"
else
    echo "Unknown Variable $VARIABLE (available: temp, prec, prec_era5_cp, prec_era5_lsp, prec_era5_sf)"
    exit 1
fi

start=$(cat $CONFIG_TEMPLATE | sed -e "s/\${NAME}/$NAME/" -e "s/\${MODEL}/$MODEL/" -e "s/\${ARCHITECTURE}/$ARCHITECTURE/" -e "s/\${LOCAL_MODULE}/$LOCAL_MODULE/" -e "s/\${GLOBAL_MODULE}/$GLOBAL_MODULE/" -e "s/\${OUTPUT_ACTIVATION}/$OUTPUT_ACTIVATION/" -e "s/\${WEIGHTED_LOSS}/$WEIGHTED_LOSS/" -e "s/\${ALPHA}/$ALPHA/" -e "s/\${MIN_YEAR}/$MIN_YEAR/" -e "s/\${MAX_YEAR}/$MAX_YEAR/" -e "s/\${MAX_TRAIN_YEAR}/$MAX_TRAIN_YEAR/" -e "s/\${MAX_VAL_YEAR}/$MAX_VAL_YEAR/" -e "s/\${N_COMPONENTS}/$N_COMPONENTS/" -e "s/\${NON_LOCAL_DIST}/$NON_LOCAL_DIST/" -e "s/\${N_PARAM_SETS}/$N_PARAM_SETS/" -e "s/\${EPOCHS}/$EPOCHS/" -e "s/\${EARLY_STOPPING}/$EARLY_STOPPING/" > $CONFIG_PATH)

if [ "$WHERE" = "kubernetes" ]
then
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
    echo "Started job $JOB_NAME in $WHERE"

elif [ "$WHERE" = "azure" ]
then
    BASE64_CONFIG=$(cat $CONFIG_PATH | base64)

    NON_LOCAL_FLAG=''
    if [ "$MODEL_IS_BASELINE" = "True" ]
    then
        # AZURE_TEMPLATE="$DIR/k8s_baseline_job_template.yml"
        echo "Running Baselines on Azure is not implemented yet!"
        exit 1
    else
        AZURE_TEMPLATE="$DIR/msaz_job_template.yml"
    fi
    DOCKER_HUB_PW=$(security find-internet-password -w -s "index.docker.io" -a steimi)
    cat $AZURE_TEMPLATE | sed -e "s/\${JOB_NAME}/$JOB_NAME/" -e "s/\${CONFIG}/$BASE64_CONFIG/" -e "s/\${DOCKER_HUB_PW}/$DOCKER_HUB_PW/" > /tmp/azure-container-$JOB_NAME.yml
    command="az container create --resource-group EU-N --file /tmp/azure-container-$JOB_NAME.yml && rm /tmp/azure-container-$JOB_NAME.yml"
    # run it in background, don't write things to stdout and keep running when this session closes
    # eval "${command}"
    eval "${command}" &>/dev/null & disown;
    echo "Started job $JOB_NAME in $WHERE"

fi
