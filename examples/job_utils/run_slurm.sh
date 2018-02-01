#!/usr/bin/env bash
source /u/kastner/.bashrc
#export THEANO_FLAGS="floatX=float32,device=cuda,force_device=True,lib.cnmem=1"
export PYTHONUNBUFFERED=1
export HOME=/u/kastner
echo Running: python "$1" on `hostname`
python -u $1
qos=${2:-unkillable}
gpu=${3:-titanx}
echo Using gpu "$gpu" with qos "$qos"
