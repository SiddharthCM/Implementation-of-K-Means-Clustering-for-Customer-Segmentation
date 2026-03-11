# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset and select the features Annual Income and Spending Score.

2.Use the Elbow method to calculate WCSS for different cluster values to find the optimal number of clusters.

3.Apply the K-Means algorithm with the chosen number of clusters to group the data points.

4.Plot the clusters and mark the centroid of each cluster on the graph.
 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: SAKTHI SABARISH P
RegisterNumber:  212225040360
*/
```
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/acer/Downloads/Mall_Customers.csv")
print(data.head())

X = data.iloc[:, [3, 4]].values

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(X)
plt.figure(figsize=(8,6))
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s=100, c='purple', label='Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s=100, c='teal', label='Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s=100, c='cyan', label='Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s=100, c='magenta', label='Cluster 5')

plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=300, c='black', label='Centroids')

plt.title('Customer Segmentation using K-Means')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

```

## Output:
![Screenshot_10-3-2026_105839_localhost](https://github.com/user-attachments/assets/b7c4ded3-d5cb-4bdd-8ad3-3dc570a96fbe)
![Screenshot_10-3-2026_105851_localhost](https://github.com/user-attachments/assets/21e7f3b2-b93d-4d47-a982-1b5cf131a0d2)
![Screenshot_10-3-2026_105918_localhost](https://github.com/user-attachments/assets/11f84101-c4b5-4e74-9c8b-8f4229dfba69)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
