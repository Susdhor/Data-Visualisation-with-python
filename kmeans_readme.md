# Describe the K-means Algorithm and klusterings method with the help of my best teacher Eva Hegner and following the resources________ Kaggle website.

"https://towardsdatascience.com/unsupervised-learning-k-means-clustering-6fd72393573c"

""https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a""

""https://towardsdatascience.com/explain-ml-in-a-simple-way-k-means-clustering-e925d019743b""

""https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html""

""https://neptune.ai/blog/clustering-algorithms""

Clustering: Clustering is the process of dividing the datasets into groups, consisting of similar data_point. Points in the same group are similar as posible. Points in different group are as dissmilar as possible. Clustering is often used unsuppervised learning technique, that is used for classification.

k_means clustering algorithm: k-means clustering algorithm whose main goal is to group similar elements or data points into a cluster.Pick the first centroid randomly,assign each data point to the closest centroid,calculate the new centroids.Repeat the process until the centroids do not change.

How dose k-means clustering work: k-means clustering algorithm is a simple algorithm that is used to group similar elements or data points into a cluster.
__Initially, a number of centroids is chosen, there are different methods for selecting the right value of k.
__Shuffle the data and initialize the centroids randomly select k_data points for centroids without replacement.
__Create new centroids by calculating the mean value of all the samples assigned to the each previous centroids.
__Randomly initialize the centroid until the centroids do not change, so the assingment of data points to the cluster isn't changning.
__K_means clustering uses the euciidean distance to calculate the distance between the data points and the centroids.

There are two most popular matrics for evaluating the performance of k-means clustering algorithm:

1. Elbow Method.
2. Silhouette Method.

The Elbow Method: The Elbow Method is a simple way to choose the optimal value of k.Its picks the range of values and take the best among them. its calculates the within-cluster sum of square (wcss) for different values of k. The best value of k is the one that minimizes the wcss. The wcss is the sum of square distances from each data point to its closest centroid.

Silhouette Method: The Silhouette Method is a measure of how well samples are clustered. It is calculated for each data point. The Silhouette Coefficient is the ratio of intra-cluster distance to the nearest cluster's inter-cluster distance. The Silhouette Coefficient is high when clusters are well separated, low when clusters are close together. The Silhouette Coefficient lies between -1 for bad clustering and +1 for good clustering.Silhouette Coefficient is zero when clusters are linearly seperable and increases towards +1 when the dataset is well separated.

Conclusion: Kmeans clustering is a unsupervised learning algorithm that is used to group similar elements or data points into a cluster.
using pca to reduce the dimension of the data the visualization of the data is better and there is no overlaping.Moreover the visualization of the data is better when the data is not linearly seperable.BY chosing the right value of k, the visualization of the data is even
more clear.
