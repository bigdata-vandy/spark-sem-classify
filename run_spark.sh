#!/usr/bin/env bash

if [ $# -ne 0 ]; then
  echo $0: "usage: ./run_spark.sh"
  exit 1
fi


echo Reading input from $input


FLAG=0
case ${FLAG} in
    (0)
        # Run application locally
        INPUT="../spark-sem-collate/data.jl"
        MASTER='local[*]'
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
