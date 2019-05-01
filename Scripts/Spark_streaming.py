## Necessary imports
from __future__ import print_function
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import sys
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt

## Create a spark context as the script is executed in batch mode/as a file
sc = SparkContext("local[*]",appName="KafkaStreaming")

## Create spark streaming context
ssc = StreamingContext(sc,1)

## Read the name of zk from the arguments 
zkQuorum,topic = sys.argv[1:]

## Listen to the topic
## Using the native Spark Streaming Kafka capabilities, we use the streaming context 
## from above to connect to our Kafka cluster. 
## The topic connected to is the topic name passed as argument arg[1:], 
## from consumer group spark-streaming-consumer. This an arbitrary name that can be changed as required.
kvs = KafkaUtils.createStream(ssc,zkQuorum,"spark-streaming-consumer",{topic:1})

## The inbound stream is a DStream, which supports various built-in transformations such as 
## map which is used here to parse the inbound messages from their native JSON format.
lines = kvs.map(lambda x : x[1])

## Function to create csv record by appending ',' between two fields
def toCSVLine(data):
    return ','.join(str(d) for d in data)

def convertDataFloat(line):
    return array([float(line[1]),float(line[2])])

savedModel = KMeansModel.load(sc, "/user/manasm/App2/kmeanModel/")

## Function to filter the noise records.


COUNT = 0
def checkRddAndWrite(rdd):
        if not rdd.isEmpty():
                global COUNT
		rdd = rdd.map(lambda line : line.split(",")).filter(lambda line : "A" in line[4])
		rdd = rdd.map(lambda line : (line[0],line[1],line[2],line[3]))
		rdd = rdd.map(toCSVLine)
                outputRdd = savedModel.predict(rdd.map(lambda line : line.split(',')).map(lambda line : convertDataFloat(line)))
                test1 = rdd.zip(outputRdd)
                path = "/user/manasm/App2/StreamData/Data" + str(COUNT)
                test1.coalesce(1).saveAsTextFile(path)
                COUNT = COUNT + 1

## Adding the filter Operations
## Writing the RDDs to the HDFS using the method checkRddAndWrite
lines.foreachRDD(lambda rdd : checkRddAndWrite(rdd))
ssc.start()
ssc.awaitTermination()
