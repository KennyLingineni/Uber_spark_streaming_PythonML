Use Case:
Uber Data.
Histroical Data Contains -> Data, Time Latitude,Longitude.
Building Geographical clusters based on Latitude and Longitude.

Code has been written in Multiple Stages for different steps/procedures.
Part 1)
send the historical data to HDFS.

Flume is to transfter the data.
We took 2% sample data -> We used Pyspark Here. These commands are executed in Jupyter note book.

Part 2)
 Here clustering we did and we arrived for k-value -  (Just Python.No spark Here)
The Model was only on 2%.

Part 3)
Using the K -value, we  built the model on 100% data. -using Spark ML(which uses RDD. Library is sprk.mllib.)
Save the Model.


