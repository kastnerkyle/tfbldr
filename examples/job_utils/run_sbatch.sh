#!/bin/bash
qos=${2:-unkillable}
gpu=${3:-titanxp}
exclude=${4:-}
force=${5:-}

if [ $# -ge 5 ]
then
    sbatch -o "slurm-%j_$1.out" --comment "$1" --get-user-env --gres=gpu:$gpu --qos=$qos --mem=56999 --exclude=$exclude --nodelist=$force --no-requeue ./run_slurm.sh $1 $2 $3
    exit
fi
    sbatch -o "slurm-%j_$1.out" --comment "$1" --get-user-env --gres=gpu:$gpu --qos=$qos --mem=56999 --exclude=$exclude --no-requeue ./run_slurm.sh $1 $2 $3

