from data_visualization import DataVisulaization


if __name__ == '__main__':
    dataclass = DataVisulaization()
    
    # Load the data set 
    dataclass.load_data('winequality-red.csv')
    
    # Missing values Check
    dataclass.missing_values()
    
    # Histogram for each feature
    dataclass.histogram()
    
    # Outliar removal for each feature
    dataclass.outliar_removal()
    
    # Skewness check for each feature
    dataclass.skewness()
    
    # Box plot for each feature
    dataclass.box_plot()
    
    # Pair plot for each feature
    dataclass.pairplot()
    
    # Scaled data for each feature
    dataclass.sklarized_data()
    
    # TSNE for each feature scaled data
    dataclass.tsne_scalar()
    
    # TSNE for each feature without scaling
    dataclass.tsne()
    
    # PCA for each feature
    dataclass.pca()
    
    # Elbow method for Kmeans clustering
    # dataclass.elbow_method()
    
    # Silhoutte score for DBSCAN clustering
    dataclass.find_silhoutte_score()
    
    # DBSCAN Clustering with scaled data
    dataclass.dbscan_clustering()
    
    # DBSCAN Clustering  with PCA
    dataclass.dbscan_pca()
    
    # TSNE for DBSCAN Clustering with PCA Data
    dataclass.tsne_dbscan()
    
    # DBSCAN Metrics Evaluation
    dataclass.dbscan_metrics_pca()
    