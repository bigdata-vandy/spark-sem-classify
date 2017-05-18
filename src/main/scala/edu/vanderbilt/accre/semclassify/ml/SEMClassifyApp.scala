package edu.vanderbilt.accre.semclassify.ml
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
  * Created by joshuaarnold on 5/18/17.
  */
object SEMClassifyApp {

  def main(args: Array[String]): Unit = {

    val input = args.length match {
      case 1 => args(0)
      case _ => throw new IllegalArgumentException(
          "Usage: SEMClassifyApp input_file_path"
        )
    }

    val conf = new SparkConf()

    val sc = new SparkContext(conf)

    val sqlContext = new SQLContext(sc)

    val data: DataFrame = sqlContext.read.format("json")
        .load(input)
        .cache()


    data.filter(!data("label").like("None") && !data("label").like(""))

    // Split the data into training and test sets (30% held out for testing)
    val Array(trainData, testData) = data.randomSplit(Array(0.7, 0.3))


    // Assemble feature columns into a single column
    val assembler =  new VectorAssembler()
        .setInputCols(Array("bse", "ca", "si", "al"))
        .setOutputCol("features")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
        .setInputCol("label")
        .setOutputCol("indexedLabel")
        .fit(data)

    // Specify a RandomForest model.
    val rf = new RandomForestClassifier()
        .setLabelCol("indexedLabel")
        .setFeaturesCol("features")
        .setNumTrees(10)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictionLabel")
        .setLabels(labelIndexer.labels)


    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
        .setStages(Array(assembler, labelIndexer, rf, labelConverter))


    // Train model.  This also runs the indexers.
    val model = pipeline.fit(trainData)

    // Evaluate the model
    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("precision")

    // Check train accuracy
    val predictionsTrain = model.transform(trainData)
    val accuracyTrain = evaluator.evaluate(predictionsTrain)

    // Check test accuracy
    val predictionsTest = model.transform(testData)
    val accuracyTest = evaluator.evaluate(predictionsTest)

    println(s"Train Accuracy: $accuracyTrain\nTest Accuracy:  $accuracyTest")

    val rfModel = model.stages(1).asInstanceOf[RandomForestClassificationModel]
    println("Learned classification forest model:\n" + rfModel.toDebugString)

  }

}
