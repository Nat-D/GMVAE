#!/bin/bash


# Switch to script directory
cd `dirname -- "$0"`

if [ -z "$1" ]; then
  echo "Please enter DataSet, e.g. ./run spiral"
  exit 0
else
  DATA=$1
  shift
fi

if [ "$DATA" == "spiral" ]; then
  th main.lua -dataSet spiral -visualise2D 1 -xSize 2 -wSize 2 -continuous 1 -hiddenSize 100 -K 8 -cvWeight 2.0 -batchSize 200 -epoch 500 -seed 1 "$@"

elif [ "$DATA" == "mnist" ]; then
  th main.lua -ACC 1 -visualGen 1 -K 10 -gpu 1 -batchSize 50 -inputDimension 2 -network conv -xSize 200 -wSize 150 "$@"
fi
