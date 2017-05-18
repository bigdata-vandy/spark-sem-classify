#!/usr/bin/env bash

case "$#" in
    (0)
        FLAG=local
        ;;
    (1)
        FLAG=$1
        ;;
    (*)
        echo $0: "usage: ./run_spark.sh [local|bigdata]"
        exit 1
        ;;
esac

echo Reading input from $input


case ${FLAG} in
    (local)
        # Run application locally
        INPUT="../spark-sem-collate/data.jl"
        MASTER='local[*]'
        ;;
    (bigdata)
        # Run application locally
        INPUT="hdfs:///user/arnoldjr/data.jl"
        MASTER='yarn'
        ;;
    (*)
        echo "Invalid FLAG option [$FLAG]"
        exit 1
        ;;
esac


spark-submit \
  --class edu.vanderbilt.accre.semclassify.ml.SEMClassifyApp \
  --master ${MASTER} \
  target/scala-2.10/spark-sem-classify_2.10-1.0.jar \
  ${INPUT}
