# Clustering

#### Unsupervised Learning


```python
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

plt.style.use('seaborn')

x, y = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=0.5, shuffle=True, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c='k', marker='+', s=50)
plt.show()
```


    
![png](images/04/output_2_0.png)
    



```python
y
```




    array([1, 2, 0, 3, 1, 0, 2, 0, 0, 0, 0, 1, 2, 2, 1, 0, 3, 3, 3, 1, 3, 2,
           1, 2, 1, 1, 3, 1, 1, 0, 3, 0, 2, 1, 2, 0, 2, 0, 0, 3, 3, 3, 3, 0,
           1, 2, 0, 3, 3, 0, 3, 2, 2, 3, 0, 3, 2, 1, 0, 1, 3, 0, 1, 0, 3, 0,
           1, 3, 2, 2, 1, 1, 0, 0, 2, 3, 2, 2, 1, 1, 3, 0, 1, 2, 2, 0, 0, 1,
           2, 2, 3, 3, 3, 0, 0, 1, 2, 3, 0, 0, 3, 1, 1, 3, 3, 2, 0, 1, 1, 0,
           3, 2, 2, 1, 1, 0, 2, 2, 3, 0, 3, 2, 0, 1, 0, 2, 0, 2, 1, 0, 2, 1,
           2, 3, 3, 2, 3, 3, 1, 3, 1, 2, 1, 1, 2, 1, 0, 1, 2, 3, 3, 3, 0, 2,
           1, 0, 3, 2, 3, 1, 3, 1, 1, 2, 2, 3, 0, 1, 1, 2, 1, 0, 2, 3, 3, 3,
           0, 0, 2, 0, 2, 1, 0, 1, 0, 2, 2, 3, 0, 3, 1, 1, 0, 3, 2, 2, 2, 0,
           3, 1])



## K-means

1- Randomly select centroids (center of cluster) for each cluster.

2- Calculate the distance of all data points to the centroids.

3- Assign data points to the closest cluster.

4- Find the new centroids of each cluster by taking the mean of all data points in the cluster.

5- Repeat steps 2,3 and 4 until all points converge and cluster centers stop moving.

<img src="images/04/kmeans.gif">

## Euclidean Distance

$$dist(A,B) = \sqrt{\sum_{i=1}^{n}{(A_i - B_i)^2}}$$

## Minkowski Distance

$$dist(A,B) = \sqrt[p]{\sum_{i=1}^{n}{(A_i - B_i)^p}}$$

* **n_clusters**: the number of desired clusters

* **n_init**: run the k-means clustering algorithms 10 times independently with different random centroids to choose the final model as the one with the lowest SSE
        
* **max_iter**: maximum number of iterations for each run

* **tol**: tolerance regarding the changes in the within-cluster SSE to declare convergence


```python
from sklearn.cluster import KMeans

#n_clusters = k
km = KMeans(n_clusters=4, init='random', n_init=10, max_iter=100, tol=1e-04, random_state=0)
y_km = km.fit_predict(x)

plt.scatter(x[y_km==0, 0], x[y_km==0, 1], s=40, c='r', marker='s', label='Cluster 1')
plt.scatter(x[y_km==1, 0], x[y_km==1, 1], s=40, c='g', marker='o', label='Cluster 2')
plt.scatter(x[y_km==2, 0], x[y_km==2, 1], s=40, c='b', marker='v', label='Cluster 3')
plt.scatter(x[y_km==3, 0], x[y_km==3, 1], s=40, c='y', marker='p', label='Cluster 4')

plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=200, marker='X', c='k', label='Centroids')

plt.legend(scatterpoints=1)
plt.show()
```


    
![png](images/04/output_12_0.png)
    


### The Elbow Method

**Theorem**: if k increases, the within-cluster SSE (“distortion”) will decrease.
    
**Inertia** is calculated by measuring the distance between each data point and its centroid.


```python
inertias = []
for i in range(1, 15):
    km = KMeans(n_clusters=i, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
    km.fit(x)
    inertias.append(km.inertia_)
    
plt.plot(range(1,15), inertias, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
```


    
![png](images/04/output_14_0.png)
    



```python
from sklearn.datasets import make_moons

x, y = make_moons(n_samples=200, noise=0.05, shuffle=True, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c='k', marker='+', s=50)
plt.show()
```


    
![png](images/04/output_15_0.png)
    



```python
from sklearn.datasets import make_circles

x, y = make_circles(n_samples=200, noise=0.03, shuffle=True, random_state=0)

plt.scatter(x[:, 0], x[:, 1], c='k', marker='+', s=50)
plt.show()
```


    
![png](images/04/output_16_0.png)
    


## DBScan

<img src="images/04/dbscan.png">


```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
newX = scaler.fit_transform(x)

dbscan = DBSCAN(eps=0.35, min_samples=3)

y_db = dbscan.fit_predict(newX)

plt.scatter(newX[:,0], newX[:,1], c=y_db, cmap='plasma')
plt.show()
```


    
![png](images/04/output_19_0.png)
    


## Hierarchical Clustering


```python
import numpy as np
import pandas as pd

df = pd.read_csv('files/Mall_Customers.csv')
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CustomerID</th>
      <th>Genre</th>
      <th>Age</th>
      <th>Annual Income (k$)</th>
      <th>Spending Score (1-100)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Male</td>
      <td>19</td>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Male</td>
      <td>21</td>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Female</td>
      <td>20</td>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Female</td>
      <td>23</td>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Female</td>
      <td>31</td>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>196</td>
      <td>Female</td>
      <td>35</td>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>197</td>
      <td>Female</td>
      <td>45</td>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>198</td>
      <td>Male</td>
      <td>32</td>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>199</td>
      <td>Male</td>
      <td>32</td>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>200</td>
      <td>Male</td>
      <td>30</td>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 5 columns</p>
</div>




```python
x = df.iloc[:, [3,4]].values
pd.DataFrame(x)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15</td>
      <td>39</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15</td>
      <td>81</td>
    </tr>
    <tr>
      <th>2</th>
      <td>16</td>
      <td>6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>77</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>40</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>195</th>
      <td>120</td>
      <td>79</td>
    </tr>
    <tr>
      <th>196</th>
      <td>126</td>
      <td>28</td>
    </tr>
    <tr>
      <th>197</th>
      <td>126</td>
      <td>74</td>
    </tr>
    <tr>
      <th>198</th>
      <td>137</td>
      <td>18</td>
    </tr>
    <tr>
      <th>199</th>
      <td>137</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
<p>200 rows × 2 columns</p>
</div>



### Single-linkage


```python
import scipy.cluster.hierarchy as sch

dendrogram = sch.dendrogram(sch.linkage(x, method='single'))

plt.title('Single')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](images/04/output_24_0.png)
    


### Complete linkage


```python
dendrogram = sch.dendrogram(sch.linkage(x, method='complete'))

plt.title('Complete')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](images/04/output_26_0.png)
    


### Average linkage


```python
dendrogram = sch.dendrogram(sch.linkage(x, method='average'))

plt.title('Average')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](images/04/output_28_0.png)
    


### Ward linkage


```python
dendrogram = sch.dendrogram(sch.linkage(x, method='ward'))

plt.title('Ward')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
```


    
![png](images/04/output_30_0.png)
    


### Agglomerative Clustering


```python
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')

y_ac = ac.fit_predict(x)

plt.scatter(x[y_ac==0, 0], x[y_ac==0, 1], s=40, c='r', label='Cluster 1')
plt.scatter(x[y_ac==1, 0], x[y_ac==1, 1], s=40, c='g', label='Cluster 2')
plt.scatter(x[y_ac==2, 0], x[y_ac==2, 1], s=40, c='b', label='Cluster 3')
plt.scatter(x[y_ac==3, 0], x[y_ac==3, 1], s=40, c='y', label='Cluster 4')
plt.scatter(x[y_ac==4, 0], x[y_ac==4, 1], s=40, c='m', label='Cluster 5')

plt.title('Agglomerative')
plt.xlabel('Annual Income')
plt.ylabel('Spendings')
plt.show()
```


    
![png](images/04/output_32_0.png)
    


http://infolab.stanford.edu/~ullman/mmds/ch7.pdf
