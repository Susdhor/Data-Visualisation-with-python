# Describe the DBSCAN algorithm and clustering method with the help of following resources______ and Kaggle website.

""https://www.kdnuggets.com/2020/04/dbscan-clustering-algorithm-machine-learning.html""
""https://towardsdatascience.com/explaining-dbscan-clustering-18eaf5c83b31""
""https://www.mygreatlearning.com/blog/dbscan-algorithm/""
""https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/""
""https://www.analyticsvidhya.com/blog/2020/09/how-dbscan-clustering-works/""
""https://medium.com/@tarammullin/dbscan-2788cfce9389""
""https://neptune.ai/blog/clustering-algorithms""

DBSCAN algorithm: DBSCAN stands for Density-Based Spatial Clustering of Applications with Noise. Its a density-based clustering algorithm that assumes that clusters are dense regions in space that are separated by regions having a lower density of data points. A unique feature of DBSCAN is that it is robust to outliers.

How the DBSCAN algorithm works: Algorithms start by picking a point (one record) x from the dataset at random and assign it to a cluster 1. Than it counts how many points are located within the epsilon distance from x. If the quantity is greater than min_points, than x is considered a core point and is added to the cluster 1. It will than examine each member of cluster 1 and find their respective e-neighbours, it will expand the cluster 1 by adding all of their e-neighbours. It will repeat this process until all points in the dataset have been assigned to a cluster. In the latter case, it will pick another point from the dataset not belonging to any cluster and assign it to a new cluster. It will continue like this until all examples either belong to a cluster or a outliers.

Core point(x): data point that has at least min_points points within epsilon distance.
Border point(y): data point that has at least one core point within epsilon distance and lower than min_points within epsilon distance from the core point.
Noise point(z): data point that has no core point within epsilon distance.

DBSCAN parameter selection:
min-points: As a starting point, a minimum n can be derived from the number of dimensions D in the dataset, as n>=D + 1.For dataset with noise larger values are usually better and will result in a more accurate clustering. For datasets with no noise, a value of 5 is usually a good starting point. 

Epsilon(€): The epsilon parameter is the maximum distance between two data points in the same cluster.If the small epsilon is chosen, a large part of the data will not be clusterd. whereas, for a too high value of €, clusters will merge and the majority of objects will be same cluster.

DBSCAN matrics: The Silhouette Coefficient is a measure of how well samples are clustered. It is calculated for each data point. The Silhouette Coefficient is the ratio of intra-cluster distance to the nearest cluster's inter-cluster distance. The Silhouette Coefficient is high when clusters are well separated, low when clusters are close together. The Silhouette Coefficient lies between -1 for bad clustering and +1 for good clustering.Silhouette Coefficient is zero when clusters are linearly seperable and increases towards +1 as the number of clusters increases.

Conclusions: DBSCAN is a good clustering algorithm for unsupervised learning. It is robust to outliers and can be used for clustering of highly unbalanced datasets. It is also a good algorithm for clustering of highly unbalanced datasets.
Density-based clustering algorithms can cluster of arbitrary shape and size. With the DNSCAN
we can clusters in dataset that exibit wide differences in density. Parameter like epsilon and min_points can be tuned to achieve the best results. However, the chose of parameters tuning is very deficut compared to other clustering algorithms as like K-means. 

In this data set i got silhouette coefficient of 0.37. This means that the clusters are well separated and the data points are well clustered.

And Davies-Bouldin score is 3-6. This means that the clusters are well separated and the data points are well clustered.
