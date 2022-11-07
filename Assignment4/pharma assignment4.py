#!/usr/bin/env python
# coding: utf-8

# In[52]:


#import the required librarys
import numpy as np 
import pandas as pd 
from sklearn import preprocessing 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt
import seaborn as sns 


# In[53]:


PM_set=pd.read_csv(r"I:\Rworkshop\Pharmaceuticals.csv")
PM_set.head()


# In[55]:


PM_set.head()


# In[56]:


PM_set.isnull().sum()


# In[57]:


PM_set1=PM_set.copy()


# In[58]:


PM_set2=PM_set1.drop(['Symbol','Name','Median_Recommendation','Location','Exchange'] ,axis=1)


# In[59]:


PM_set2.head()


# In[60]:


#Normalising the data
dfnormalize= preprocessing.normalize(PM_set2)
print(type(PM_set2))
print(PM_set2.columns)
PM_clustering=pd.DataFrame(dfnormalize, columns=PM_set2.columns)
PM_clustering.head()


# In[80]:


#PM_clustering["clusters"] = kmeans.labels_
#print(data_for_clustering["clusters"])
# Inertia: Sum of distances of samples to their closest cluster center
sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(PM_clustering)
    
    sse[k] = kmeans.inertia_ 
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# In[64]:


# From the graph we can say that there are 3 clusters
kmeans = KMeans(n_clusters=3) 
kmeans.fit(PM_clustering)
cluster_ids=kmeans.predict( PM_clustering)
cluster_ids


# In[65]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(PM_clustering)
means=kmeans.cluster_centers_
print(means)


# In[66]:


plt.scatter(PM_clustering.Market_Cap[cluster_ids==0],PM_clustering.Beta[cluster_ids==0],c='purple',label='cluster0')
plt.scatter(PM_clustering.Market_Cap[cluster_ids==1],PM_clustering.Beta[cluster_ids==1],c='black',label='cluster1')
plt.scatter(PM_clustering.Market_Cap[cluster_ids==2],PM_clustering.Beta[cluster_ids==2],c='green',label='cluster2')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],marker='+',c='red',s=250,label='centroids')
plt.title('market_cap vs beta ')
plt.show()


# In[84]:


import sklearn.metrics as metrics
import sklearn.cluster as cluster
SK = range(3,13)
sil_score = []
for i in SK:
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=42).fit(PM_clustering).labels_
    score = metrics.silhouette_score(PM_clustering,labels,metric="euclidean",sample_size=250,random_state=42)
    sil_score.append(score)
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(PM_clustering,labels,metric="euclidean",sample_size=250,random_state=42)))


# In[85]:


sil_centers = pd.DataFrame({'Clusters' : SK, 'Sil Score' : sil_score})
sil_centers


# In[86]:


sns.lineplot(x = 'Clusters', y = 'Sil Score', data = sil_centers, marker="+")


# In[96]:


PM_clustering['Clusters'] = kmeans.labels_


# In[97]:


sns.scatterplot(x="Market_Cap", y="Beta",hue = 'Clusters',  data=PM_clustering)


# In[98]:


# BY observing two method I was choosing k=3


# In[99]:


PM_clustering[cluster_ids==0].describe()


# In[100]:


PM_clustering[cluster_ids==1].describe()


# In[101]:


PM_clustering[cluster_ids==2].describe()


# In[ ]:


# Based on comparing in Market_cap the three cluster standard deviation and and mean vlaues we can say that companies are devided into   cluster 0 are large, cluster 1 are medium and cluster 2 small;
# In Beta column it has  high mean values in cluster 0 
# In PE_Ratio cluster 0 have high value , and cluster 1 has very low and cluster 2 has medium value it has high growth
# By the mean and std value in ROE cluster 2 has high value ,and  other to have low values and cluster 0 have less value.
#In ROA Cluster 0 having low mean value with respect to other 2 clusters.
# Asset_Turnover cluster1 haveing high mean value it generating more revenue on investement
#Leverage mean debt to investing on companies cluster 1 have more debt mean vlaue compared to other
# Rev_Growth cluster 2 having highest value then other cluster 1 have least mean value
# Net_Profit_Margin cluster2 have highest net margin and cluster 0 having lowest net profit


# In[ ]:


#Pattern in numerical variables


# In[93]:


PM_set[cluster_ids==0]


# In[ ]:


#In this table of cluster 0 all companies in NYSE, having Median_Recommendation hold and moderate buy


# In[94]:


PM_set[cluster_ids==1]


# In[ ]:


#In this table of cluster1  Diverse companies in NYSE and majority are in US,UK. having Median_Recommendation hold and moderate buy


# In[95]:


PM_set[cluster_ids==2]


# In[ ]:


#In this table of cluster 2 Diverse companies in NYSE,NASDAQ and AMEX, having Median_Recommendation hold,moderate sell and moderate buy.
#These are majorly located in US,UK and Ireland


# In[ ]:


#NAMES
#Business invloves finance and investement to run so i would like to name them as Risk
# It as High,Medium and Low 
# From the variable i can say
# cluster 0 LOW RISK (because it has good market cap and better lavarage value )
# Cluster2 MEDIUM RISK(it has good economics growth, net profit and less laverage)
# cluster1 HIGH RISK(It has more laverage than turnover )

