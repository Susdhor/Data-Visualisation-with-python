from dataclasses import dataclass
import dataclasses
from turtle import color, width
from cv2 import PCA_DATA_AS_COL
from matplotlib import markers, pyplot as plt
import numpy as np
import pandas as pd
from regex import F
import seaborn as sns
from sklearn import metrics
from tqdm import tqdm
from sklearn.manifold import TSNE


from sklearn.cluster import KMeans, k_means
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error 


import warnings
warnings.filterwarnings('ignore')

# craete a class for K-means

class DataVisulaization:
    
      # Empty constructor
    def __init__(self):
        """DataVisulaization constructor."""
    pass
     

    # Function  for load the data from csv file and checking shape of data,data type 
    # and statistical summary of data
    def load_data(self, file_name):
        """Load data from a csv file.Function  for load the data from csv file and 
           checking shape of data,data type and statistical summary of data"""
        self.data = pd.read_csv(file_name)
        print(self.data.head())
        print(f'------------------------------------------------------')
        print(self.data.info())
        print(f'------------------------------------------------------')
        print(f'Shape of Dataset: {self.data.shape}')
        print(f'------------------------------------------------------')
        print(f'Datatypes of each column: {self.data.dtypes}')
        print(f'------------------------------------------------------')
        print(f'data describre: {self.data.describe()}')
    

        
    # function for checking missing values
    def missing_values(self):
        """Missing values of the data. how many missing values in each column."""
        total = self.data.isnull().sum().sort_values(ascending=False)
        percent = (self.data.isnull().sum()/self.data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(f'------------------------------------------------------')
        print(f'Total Missing Value: {missing_data.head(20)}')
        
    # function for plotting histogram of each column, 
    def histogram(self):
        """Histogram of each column, function for plotting histogram of each column,
        the main idea here is to visualize the data distribution of each column 
        * check the kind of each feature distribution, check data symmetry,  verify
        features frequency, identify outliers."""
        for x in self.data.columns:
            self.data.hist(column=x, figsize=(5,5), bins=50, xlabelsize=5, ylabelsize=5)
            plt.show()
            
    # function for outliar removal by using IQR method
    def outliar_removal(self):
        """Outliar removal by using IQR method"""
        for x in self.data.columns:
            Q1 = self.data[x].quantile(0.25)
            Q3 = self.data[x].quantile(0.75)
            IQR = Q3 - Q1
            self.data = self.data[~((self.data[x] < (Q1 - 1.5 * IQR)) | (self.data[x] > (Q3 + 1.5 * IQR))).values]
            print(f'------------------------------------------------------')
            print(f'Outliar Removed for {x}')
            
            
    # function for checking the skewness of dataset
    def skewness(self):
        """ checking the skewness of dataset, skewness is a measure of the asymmetry of"""
        for x in self.data.columns:
            print(f'------------------------------------------------------')
            print(f'Skewness of {x} is {self.data[x].skew()}')
    
            
    # function for box plot of each column
    def box_plot(self):
        """function for box plot of each column.A box plot is a method for graphically
        depicting groups of numerical data through their quartiles. The box extends from
        the Q1 to Q3 quartile values of the data, with a line at the median (Q2). 
        The whiskers extend from the edges of box to show the range of the data. 
        The position of the whiskers is set by default to 1.5*IQR (IQR = Q3 - Q1)from 
        the edges of the box. Outlier points are those past the end of the whiskers."""
        for x in self.data.columns:
            self.data.boxplot(column=x, figsize=(5,5))
            plt.show()
            
    # function for scatter plot for all columns
    def pairplot(self):
        """function for scatter plot for all columns, the main idea here is to visualize
        the data distribution of each column, check the kind of each feature distribution,
        """
        sns.pairplot(self.data, size=1.5)
        plt.title('Pair Plot')
        plt.show()
        
              
    # function for sklarized data by using StandardScaler
    def sklarized_data(self):
        """function for sklarized data by using StandardScaler, the main idea here is to
        the data to have a mean of 0 and a variance of 1. This is called normalization,
        by using StandardScaler we can normalize the data for better performance of 
        clustering algorithms."""
        self.data_scalar = StandardScaler().fit_transform(self.data)
        print(f' Standard Scaler data : {self.data_scalar}')
        
        
    # function for tsne with all variables scaled data
    def tsne_scalar(self):
        """function for tsne with all variables, the main idea here is to visualize
        the data distribution and visualization of high dimensional data."""
        tsne = TSNE(random_state=42,perplexity=30)
        tsne_results = tsne.fit_transform(self.data_scalar)
        sns.scatterplot(x=tsne_results[:,0], y=tsne_results[:,1], data=self.data_scalar)
        plt.title('TSNE SCALAR DATA')
        #plt.xlabel('t-SNE 1')
        #plt.ylabel('t-SNE 2')
        
        plt.show()
    # function for tsne with all variables without scaled data    
    def tsne(self): 
        """function for tsne with all variables, the main idea here is to visualize
        the data distribution and visualization of high dimensional data""" 
        tsne = TSNE(n_components=2, random_state=42, perplexity=7)
        tsne_data = tsne.fit_transform(self.data)
        tsne_df = pd.DataFrame(data = tsne_data
             , columns = ['t-SNE 1', 't-SNE 2'])
        print(f'TSNE data : {tsne_df}')
        sns.scatterplot(x='t-SNE 1', y='t-SNE 2', data=tsne_df)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title('t-SNE without Scaled Data:: ', fontsize=16)
        plt.show()

        
        
        
    # function for PCA
    def pca(self):
        """function for PCA, the main idea here is to reduce the unnecessary feature
        and dimension and make it easier to visualize the data, by using PCA we can
        lose some information and variance in the data."""
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(self.data_scalar)
        principalDf = pd.DataFrame(data = pca_data
             , columns = ['principal component 1', 'principal component 2'])
        print(f'PCA data : {principalDf}')
        print(f'Cumulative explained variance ratio : {np.sum(pca.explained_variance_ratio_)}')        
        
        sns.scatterplot(x='principal component 1', y='principal component 2', data=principalDf)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('2 component PCA', fontsize=16)
        plt.show()
        
           
    # function for finding elbow method
    def elbow_method(self):
        """function for finding elbow method, the main idea here is to find the optimal
        number of clusters by using elbow method, by using elbow method we can find
        the optimal number of clusters."""
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i)
            kmeans.fit(self.data_scalar)
            inertia.append(kmeans.inertia_)
        print(f'Inertia value : {inertia}')
        sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
        plt.subplot(1,2,2)
        plt.plot(range(1, 11), inertia,markersize=10,marker='o',color='red',linewidth=3)
        plt.title('Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('Inertia',fontweight='bold')
        plt.grid(True)
        plt.show()
        
        
    # function for silhourtte score
    def find_silhoutte_score(self):
        """function for silhourtte score, the main idea here is to find the optimal
        number of clusters by using silhourtte score, by using silhourtte score we can
         find the value k that gives the best silhouette score."""
        silhouette = {}
        for i in range(2, 4):
            k_means = KMeans(n_clusters=i,
                             init='k-means++',
                             n_init=15,
                             max_iter=500,
                             random_state=17)
            k_means.fit(self.data_scalar)
            silhouette[i] = silhouette_score(self.data_scalar, k_means.labels_, metric='euclidean')
            print(f'Silhoutte score for {i} clusters : {silhouette[i]}')
            plt.subplot(2, 2, i)
            plt.bar(range(len(silhouette)), list(silhouette.values()), align='center',color= 'red',width=0.5)
            plt.xticks(range(len(silhouette)), list(silhouette.keys()))
            plt.grid()
            plt.title('Silhouette Score',fontweight='bold')
            plt.xlabel('Number of Clusters')
            plt.show()
        
    # function for kmeans clustering using scaled data
    
    def kmeans_clustering(self):
        """function for kmeans clustering, the main idea here is to group the data
        into clusters by using kmeans clustering, by using kmeans clustering we can
        group the data into clusters.""" 
        self.kmeans = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.kmeans.fit(self.data_scalar)
        print(f'Kmeans clustering : {self.kmeans.labels_}')
        print(f'Kmeans clustering : {self.kmeans.cluster_centers_}')
        print(f'Kmeans clustering : {self.kmeans.inertia_}')
        
        sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
        plt.scatter(self.data_scalar[:,0], self.data_scalar[:,1], c=self.kmeans.labels_, cmap='rainbow')
        plt.scatter(self.kmeans.cluster_centers_[:,0], self.kmeans.cluster_centers_[:,1], color='black', marker='x', s=150, linewidths=5)
        #plt.xlabel('Principal Component 1', fontsize=12)
        #plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('Kmeans Clustering with scaled data: ', fontsize=16)
        plt.show()

    # PCA Kmeans clustering
    def pca_clustering(self):
        """function for PCA clustering, the main idea here is to use principal
        components data to group the data into cluster by using kmeans clustering."""
        self.pca = PCA(n_components=2)
        self.pca_data = self.pca.fit_transform(self.data_scalar)
        self.principalDf = pd.DataFrame(data = self.pca_data
                , columns = ['principal component 1', 'principal component 2']) 
        self.pca_clustering = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.pca_clustering.fit(self.principalDf)
        sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
        plt.scatter(self.principalDf['principal component 1'], self.principalDf['principal component 2'], c=self.pca_clustering.labels_, cmap='rainbow')
        plt.scatter(self.pca_clustering.cluster_centers_[:,0], self.pca_clustering.cluster_centers_[:,1], color='black', marker='x', s=150, linewidths=5)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('PCA Clustering with PCA data: ', fontsize=16)
        plt.show()    
    # function for using dbscan clustering on sklarized data
    def dbscan_clustering(self):
        """function for dbscan clustering, the main idea here is to group the data 
         into clusters by using dbscan clustering."""
        self.dbscan = DBSCAN(eps=0.5, min_samples=5).fit(self.data_scalar)
        print(f'DBSCAN clustering : {self.dbscan.labels_}')
        print(f'DBSCAN clustering : {self.dbscan.core_sample_indices_}')
        print(f'DBSCAN clustering : {self.dbscan.components_}')
        
        sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
        plt.scatter(self.data_scalar[:,0], self.data_scalar[:,1], c=self.dbscan.labels_, cmap='rainbow')
        plt.scatter(self.dbscan.components_[:,0], self.dbscan.components_[:,1], color='black', marker='x', s=150, linewidths=5)
        #plt.xlabel('Principal Component 1', fontsize=12)
        #plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('DBSCAN Clustering scaler DATA: ', fontsize=16)
        plt.show()        
        
        
        
    # function to visualize pca with tsne
    def pca_tsne(self):
        """function for pca tsne, the main idea here is to use principal
        components data to group the data into cluster by using tsne clustering."""
        self.pca = PCA(n_components=2)
        self.pca_data = self.pca.fit_transform(self.data_scalar)
        self.principalDf = pd.DataFrame(data = self.pca_data
                , columns = ['principal component 1', 'principal component 2']) 
        self.pca_clustering = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.pca_clustering.fit(self.principalDf)
        tsne = TSNE(n_components=2, random_state=0, perplexity=7)
        tsne_data = tsne.fit_transform(self.pca_data)
        tsne_df = pd.DataFrame(data = tsne_data
             , columns = ['t-SNE 1', 't-SNE 2'])
        print(f'TSNE data : {tsne_df}')
        sns.scatterplot(x='t-SNE 1', y='t-SNE 2', data=tsne_df)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title('TSNE PCA DATA', fontsize=16)
        plt.show()
        
    # function for find out epsilon value for dbscan with pca data
    def dbscan_pca(self):
        """function for dbscan clustering with pca data, the main idea here is to group the data
            into clusters by using dbscan clustering."""
        
        self.pca = PCA(n_components=2)
        self.pca_data = self.pca.fit_transform(self.data_scalar)
        self.principalDf = pd.DataFrame(data = self.pca_data
                , columns = ['principal component 1', 'principal component 2']) 
        self.pca_clustering = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.pca_clustering.fit(self.principalDf)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5).fit(self.pca_data)
        print(f'DBSCAN clustering : {self.dbscan.labels_}')
        print(f'DBSCAN clustering : {self.dbscan.core_sample_indices_}')
        print(f'DBSCAN clustering : {self.dbscan.components_}')
        
        sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
        plt.scatter(self.pca_data[:,0], self.pca_data[:,1], c=self.dbscan.labels_, cmap='rainbow')
        plt.scatter(self.dbscan.components_[:,0], self.dbscan.components_[:,1], color='black', marker='x', s=150, linewidths=5)
        plt.xlabel('Principal Component 1', fontsize=12)
        plt.ylabel('Principal Component 2', fontsize=12)
        plt.title('DBSCAN Clustering with PCA:', fontsize=16)
        plt.show()
        
        # find out epsilon value for dbscan with pca data
        epsilon_list = []
        for i in range(1,10):
            epsilon_list.append(i/10)
            
        for epsilon in epsilon_list:
            self.dbscan = DBSCAN(eps=epsilon, min_samples=5).fit(self.pca_data)
            print(f'DBSCAN clustering : {self.dbscan.labels_}')
            print(f'DBSCAN clustering : {self.dbscan.core_sample_indices_}')
            print(f'DBSCAN clustering : {self.dbscan.components_}')
            
            sns.set(style="darkgrid",font_scale=1.5,rc={'figure.figsize':(11.7,8.27)})
            plt.scatter(self.pca_data[:,0], self.pca_data[:,1], c=self.dbscan.labels_, cmap='rainbow')
            plt.scatter(self.dbscan.components_[:,0], self.dbscan.components_[:,1], color='black', marker='x', s=150, linewidths=5)
            plt.xlabel('Principal Component 1', fontsize=12)
            plt.ylabel('Principal Component 2', fontsize=12)
            plt.title('PCA Clustering Epsilon', fontsize=16)
            plt.show()
            
    # function for tsne with dbscan
    
    def tsne_dbscan(self):
        """function for tsne with dbscan, the main idea here is to use tsne data to group the data
            into cluster by using dbscan clustering."""
        
        self.pca = PCA(n_components=2)
        self.pca_data = self.pca.fit_transform(self.data_scalar)
        self.principalDf = pd.DataFrame(data = self.pca_data
                , columns = ['principal component 1', 'principal component 2']) 
        self.pca_clustering = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.pca_clustering.fit(self.principalDf)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5).fit(self.pca_data)
        print(f'DBSCAN clustering : {self.dbscan.labels_}')
        print(f'DBSCAN clustering : {self.dbscan.core_sample_indices_}')
        print(f'DBSCAN clustering : {self.dbscan.components_}')
        
        tsne = TSNE(n_components=2, random_state=0, perplexity=7)
        tsne_data = tsne.fit_transform(self.pca_data)
        tsne_df = pd.DataFrame(data = tsne_data
             , columns = ['t-SNE 1', 't-SNE 2'])
        print(f'TSNE data : {tsne_df}')
        sns.scatterplot(x='t-SNE 1', y='t-SNE 2', data=tsne_df)
        plt.xlabel('t-SNE 1', fontsize=12)
        plt.ylabel('t-SNE 2', fontsize=12)
        plt.title('T-SNE DBSCAN', fontsize=16)
        plt.show()
        
    # function for DBSCAN Metrics with pca data
    def dbscan_metrics_pca(self):
        """function for dbscan metrics with pca data, the main idea here is to calculate the
            metrics for dbscan clustering."""
        
        self.pca = PCA(n_components=2)
        self.pca_data = self.pca.fit_transform(self.data_scalar)
        self.principalDf = pd.DataFrame(data = self.pca_data
                , columns = ['principal component 1', 'principal component 2']) 
        self.pca_clustering = KMeans(n_clusters=3, init='k-means++', n_init=15, max_iter=500, random_state=17)
        self.pca_clustering.fit(self.principalDf)
        self.dbscan = DBSCAN(eps=0.5, min_samples=5).fit(self.pca_data)
        print(f'DBSCAN Labels : {self.dbscan.labels_}')
        print(f'DBSCAN Core Sample : {self.dbscan.core_sample_indices_}')
        print(f'DBSCAN Components : {self.dbscan.components_}')
        
        # metrics for dbscan with pca data
        print(f'Silhouette Score : {metrics.silhouette_score(self.pca_data, self.dbscan.labels_)}')
        print(f'Davies Bouldin Score : {metrics.davies_bouldin_score(self.pca_data, self.dbscan.labels_)}')
        
      
            
        
            
            
            
        
        
    
        
     