
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy as sp
from sklearn.cluster import KMeans
get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# In[4]:


sampleData = pd.read_csv("/home/rameshm/Uber/SampleData/part-00000",  names = ['date', 'latitude', 'longitude', 'baseLLC'])


# In[5]:


sampleData.head(4)


# In[7]:


columns = ['longitude', 'latitude']
featuresData = pd.DataFrame(sampleData, columns=columns)


# In[9]:


### For the purposes of this example, we store feature data from our
### dataframe `featuresData`, in the `f1` and `f2` arrays. We combine this into
### a feature matrix `X` before entering it into the algorithm.
f1 = featuresData['longitude'].values
f2 = featuresData['latitude'].values
X=np.matrix(zip(f1,f2))


# In[10]:


X=X[:100]


# In[12]:


X


# In[13]:


## Draw a scatter plot with above features
plt.scatter(f1,f2)
plt.show()


# In[14]:


K = range(1,20)
KM = [KMeans(n_clusters=k).fit(X) for k in K]
centroids = [k.cluster_centers_ for k in KM]


# In[15]:


## Find with in sum of squared error
from scipy.spatial.distance import cdist, pdist

D_k = [cdist(X, cent, 'euclidean') for cent in centroids]
#cIdx = [np.argmin(D,axis=1) for D in D_k]
dist = [np.min(D,axis=1) for D in D_k]
sumWithinSS = [sum(d) for d in dist]


# In[8]:


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, sumWithinSS, 'b*-')
ax.plot(K, sumWithinSS, marker='o', markersize=12, 
markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('within-cluster sum of squares')
plt.title('Elbow for KMeans clustering')


# In[16]:


sumWithinSS

