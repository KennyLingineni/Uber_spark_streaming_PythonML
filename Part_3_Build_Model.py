
# coding: utf-8

# In[1]:


import os
import sys
os.environ["SPARK_HOME"] = "/usr/hdp/current/spark2-client"
os.environ["PYLIB"] = os.environ["SPARK_HOME"] + "/python/lib"
sys.path.insert(0, os.environ["PYLIB"] + "/py4j-0.10.4-src.zip")
sys.path.insert(0, os.environ["PYLIB"] + "/pyspark.zip")


## Create SparkContext, SparkSession
from pyspark.sql import SparkSession
from pyspark import SparkContext
sc = SparkContext()


# In[2]:


from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt


# In[3]:


data = sc.textFile("/user/rameshm/uber/HistData/FlumeData.*")


# In[4]:


data.count()


# In[5]:


data.take(2)


# In[6]:


def convertDataFloat(line):
    return array([float(line[1]),float(line[2])])


# In[7]:


fea_data = data.map(lambda data:data.split(','))
parsedData = fea_data.map(lambda line : convertDataFloat(line))


# In[8]:


parsedData.take(5)


# In[9]:


clusters = KMeans.train(parsedData,8, maxIterations=10, initializationMode="random")


# In[10]:


clusters.centers


# In[11]:


def wsssError(point):
    center = clusters.centers[clusters.predict(point)]
    return sqrt(sum([x**2 for x in (point - center)]))


# In[12]:


WSSSE = parsedData.map(lambda point: wsssError(point)).reduce(lambda x, y: x + y)
print("Within Set Sum of Squared Error = " + str(WSSSE))


# In[13]:


clusters.predict(array([40.6988701 , -74.20341933]))


# In[14]:


sqrt(sum((array([40.7204,-74.0047]) - array([ 40.71743048, -74.002436  ])) ** 2))


# In[14]:


#40.7204 - 40.71743048


# In[15]:


#-74.0047 - -74.002436


# In[15]:


clusters.save(sc, "/user/rameshm/Uber/kmeanModel")

