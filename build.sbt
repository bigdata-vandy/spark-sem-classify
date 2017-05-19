name := "spark-sem-classify"

version := "1.0"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "2.0.0",
  "org.apache.spark" % "spark-sql_2.10" % "2.0.0",
  "org.apache.spark" % "spark-mllib_2.10" % "2.0.0"
)