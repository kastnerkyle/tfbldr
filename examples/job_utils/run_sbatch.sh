#!/bin/bash
qos=${2:-unkillable}
gpu=${3:-titanxp}
exclude=${4:-}

sbatch -o "slurm-%j_$1.out" --comment "$1" --get-user-env --gres=gpu:$gpu --qos=$qos --mem=47999 --exclude=$exclude --no-requeue ./run_slurm.sh $1 $2 $3
