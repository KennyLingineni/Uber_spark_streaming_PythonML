# Uber_spark_streaming_PythonML
Uber_usecase using spark and streaming data in Python ML
Use Case:
Uber Data.
Histroical Data Contains -> Data, Time Latitude,Longitude.
Building Geographical clusters based on Latitude and Longitude.

Code has been written in Multiple Stages for different steps/procedures.
Part 1)
send the historical data to HDFS.

Flume is to transfter the data.
took 2% sample data -> We used Pyspark Here. These commands are executed in Jupyter note book.

Part 2)
 Here clustering is been done to arrived for the k-value -  (Just Python.No spark Here)
The Model was only on 2%.

Part 3)
Using the K -value, model is being built on 100% data. -using Spark ML(which uses RDD. Library is sprk.mllib.)
Save the Model.

