name := "spark-sem-classify"

version := "1.0"

scalaVersion := "2.10.6"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.6.2",
  "org.apache.spark" % "spark-mllib_2.10" % "1.6.2"
)