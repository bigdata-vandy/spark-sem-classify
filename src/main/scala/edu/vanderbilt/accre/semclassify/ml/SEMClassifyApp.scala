package edu.vanderbilt.accre.semclassify.ml
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SparkSession}


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

    val spark = SparkSession
        .builder()
        .appName("Classify SEM images with Spark")
        .getOrCreate()

    // For implicit conversions like converting RDDs to DataFrames
    import spark.implicits._

    val allData: DataFrame = spark.read.json(input)

    // Select the manually-labeled data, leaving out soi_011 for testing
    val data = allData.filter(
      !allData("soi").like("%soi_011") &&
          !allData("label").like("None") &&
          !allData("label").like("")
    ).cache()

    // Pick up soi_001 for testing
    val testData = allData.filter(
      allData("soi").like("%soi_011") &&
          !allData("label").like("None") &&
          !allData("label").like("")
    )

    // Split the data into training and validation sets (30% held out for testing)
    def stratifiedSample(df: DataFrame, col: String,
                         splits: Array[Double]): Array[DataFrame] = {
      val distinctLabels =
        df.select(df(col)).distinct.collect.map{case Row(s: String) => s}

      val emptyDF = spark.createDataFrame(sc.emptyRDD[Row], data.schema)
      distinctLabels
          .foldLeft(Array.fill(2)(emptyDF)){
            case (a: Array[DataFrame], likeVal: String) => {
              val b = df.filter(df(col).like(likeVal)).randomSplit(splits, 42L)
              (a zip b).map { case (a1, b1) => a1.unionAll(b1) }
            }
          }.map(df1 =>
              df1.sample(withReplacement = false, fraction = 1.0, seed = 42L)
      )
    }

    val Array(trainData, validData) =
      stratifiedSample(data, "label", Array(0.7, 0.3))


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
        .setNumTrees(30)

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
        .setInputCol("prediction")
        .setOutputCol("predictionLabel")
        .setLabels(labelIndexer.labels)


    // Chain indexers and forest in a Pipeline
    val pipeline = new Pipeline()
        .setStages(Array(assembler, labelIndexer, rf, labelConverter))

    // We use a ParamGridBuilder to construct a grid of parameters to search over.
    // With 3 values for hashingTF.numFeatures and 2 values for lr.regParam,
    // this grid will have 3 x 2 = 6 parameter settings for CrossValidator to choose from.
    val paramGrid = new ParamGridBuilder()
        .addGrid(rf.numTrees, Array(10, 30, 100))
        .build()

    // Create evaluator
    val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol("indexedLabel")
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

    // We now treat the Pipeline as an Estimator, wrapping it in a CrossValidator instance.
    // This will allow us to jointly choose parameters for all Pipeline stages.
    // A CrossValidator requires an Estimator, a set of Estimator ParamMaps, and an Evaluator.
    // Note that the evaluator here is a BinaryClassificationEvaluator and its default metric
    // is areaUnderROC.
    val cv = new CrossValidator()
        .setEstimator(pipeline)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(paramGrid)
        .setNumFolds(3)  // Use 3+ in practice

    // Train model.  This also runs the indexers.
    val cvModel = cv.fit(trainData)


    // Check train accuracy
    val predictionsTrain = cvModel.transform(trainData)
    val accuracyTrain = evaluator.evaluate(predictionsTrain)

    // Check validation accuracy
    val predictionsValid = cvModel.transform(validData)
    val accuracyValid = evaluator.evaluate(predictionsValid)

    // Check test accuracy
    val predictionsTest= cvModel.transform(testData)
    val accuracyTest = evaluator.evaluate(predictionsTest)


    println(s"Train Accuracy: $accuracyTrain\n" +
        s"Valid Accuracy: $accuracyValid\n" +
        s"Test Accuracy:  $accuracyTest")

    // val rfModel = model.stages(2).asInstanceOf[RandomForestClassificationModel]
    // println("Learned classification forest model:\n" + rfModel.toDebugString)

  }

}
