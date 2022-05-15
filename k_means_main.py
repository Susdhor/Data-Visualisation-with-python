from data_visualization import DataVisulaization

if __name__ == '__main__':
    dataclass = DataVisulaization()
    # Load the data set
    dataclass.load_data('wine-clustering.csv')
    
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
    dataclass.elbow_method()
    
    # Silhoutte score for Kmeans clustering
    dataclass.find_silhoutte_score()
    
    # Kmeans clustering with certain number of clusters
    dataclass.kmeans_clustering()
    
    # Kmeans clustering with PCA data
    dataclass.pca_clustering()
    
    # TSNE with PCA data 
    dataclass.pca_tsne()
 
    