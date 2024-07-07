
from ucimlrepo import fetch_ucirepo 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# fetch dataset 
heart_disease = fetch_ucirepo(id=45) 
  
# data (as pandas dataframes) 
x = heart_disease.data.features 
y = heart_disease.data.targets  
# variable information
print(heart_disease.variables)
print(x)


x.hist(figsize=(20, 15))
plt.suptitle('Feature Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(y, kde=True, stat="density", linewidth=0)
plt.title('Target Distribution with Histogram and KDE')
plt.show()

# %%
# Scatter Plot
plt.figure(figsize=(10, 6))
plt.scatter(x['age'], y)
plt.title('Age vs. prevalence of heart disease')
plt.xlabel('Age')
plt.ylabel('Target')
plt.show()

# %% [markdown]
# # 2.0 Data preprocessing and encoding

# %% [markdown]
# ## 2.1 Handling missing values

# %%
print(type(x))
missing_values_count = x.isnull().sum()
print(missing_values_count)

# %%
#there are 6rows in total with null values (4 ca and 2 thal), replace null values with the median of the row
x = x.fillna(x.median())
missing_val = x.isnull().sum()
print(missing_val)

# %% [markdown]
# ## 2.2 Encoding categorical values

# %%
categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang']
x_encoded = pd.get_dummies(x, columns=categorical_cols)
print(x_encoded)

# %% [markdown]
# ## 2.3 Scale Features

# %%
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_encoded)
x_scaled_df = pd.DataFrame(x_scaled, columns=x_encoded.columns)
print(x_scaled_df)


# %% [markdown]
# # 3.0 Clustering

# %% [markdown]
# ## 3.1 K-means clustering
# - The implementation here starts with k=3 for low, medium and high since the aim is to identify patients with a high risk of developing heart disease. This will later be refined usin the Elbow method or silhouette score.

# %%
from sklearn.cluster import KMeans
k = 5 
kmeans = KMeans(n_clusters=k, random_state=42)
#fit kmeans model to x_scaled
kmeans.fit(x_scaled)
# assign cluster to each datapoint
clusters = kmeans.predict(x_scaled)

#use elbow method to evaluate clustering

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow analyis')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# %%
#having adjusted the clusters from 3 to five, analyze the centroids
centroids = kmeans.cluster_centers_
features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca', 'thal',
       'sex_0', 'sex_1', 'cp_1', 'cp_2', 'cp_3', 'cp_4', 'fbs_0', 'fbs_1',
       'restecg_0', 'restecg_1', 'restecg_2', 'exang_0', 'exang_1']
centroid_df = pd.DataFrame(centroids, columns=features)
print(centroid_df)

# %% [markdown]
# ### 3.1.1 K-means clustering PCA Visualization

# %%
#visualize kmeans cluster using PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

x_pca = pca.fit_transform(x_scaled)

plt.figure(figsize=(8,6))

for i in range(k):
    plt.scatter(x_pca[clusters == i, 0], x_pca[clusters == i, 1], label=f'cluster{i}')
plt.legend()
plt.title('Clusters Visualization')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# %% [markdown]
# ### 3.1.2 K-means silhouette and Davies-Bouldin index.

# %%
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
K_davies = davies_bouldin_score(x_scaled, clusters)
k_silhouette = silhouette_score(x_scaled, clusters)
print(f'kmeans davies score is {K_davies}')
print(f'kmeans silhouette score is {k_silhouette}')

# %% [markdown]
# ## 3.2  Hierarchical Clustering
# 
# - Key concepts to understand here include:
#     - *linkage matrix*: key output of hierachichal clustering algs. Contains information about pairwise distances between elements/clusters and the sequence in which they are merged.
#     - *Dendogram* : constructed from the linkage matrix. It is a tree diagram that visually represents the process of cluster merging. Relevant in visualizing the hierachy of structure which then helps in determining the appropriate number of clusters.

# %%
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

#linkage matrix 

Z = linkage(x_scaled, method='ward')

#dendrogram

plt.figure(figsize = (20, 10))
plt.title("Dendrogram")
dendrogram(Z)
plt.axhline(y=25, color='r', linestyle='--')
plt.show()

# %% [markdown]
# ### 3.2.1 Dendrogram interpretation
# - Threshold line drawn at 25 
# - clusters determined at this threshold = 6

# %% [markdown]
# ### 3.2.2 Agglomerative clustering

# %%
n_clusters = 6
cluster = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
cluster_labels = cluster.fit_predict(x_scaled)


# %% [markdown]
# ### 3.2.3 Cluster Analysis

# %%
plt.figure(figsize=(10, 8))
plt.scatter(x_scaled[:, 0], x_scaled[:, 1], c=cluster_labels, cmap='rainbow', alpha=0.7, edgecolors='b')
plt.title('Clusters Visualization')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# %%
silhouette_avg = silhouette_score(x_scaled, cluster_labels)
print(f"Silhouette Score: {silhouette_avg:.3f}")

# %% [markdown]
# ### 3.2.4 Observations and Ammendments
# 
# - After printing the silhouette score with the initially determined clusters (*see dendrogram above*), the silhouette score is determined to be 0.141
# - This score is relatively poor which means either of two things:
#     - The data does not naturally cluster well with the chosen parameters
#     - The hierachichal clustering algorithm above and its settings are not optimal for the dataset herein
# - To improve the clustering, the follwowing approaches were used: 
#     - Ammended the number of clusters *n_clusters* using the iteration loop below. 
#     - A function **def find_best_linkage** that iterates between different linkages to determine the linkage that produces the best perfoming linkage matrix (see code below). The best perfoming number of clusters in the previous method is used as the optimal when calling this function

# %%
best_score = -1
best_n_clusters = 0

for n_clusters in range(2, 11):  # Trying different numbers of clusters
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    cluster_labels = clusterer.fit_predict(x_scaled)
    silhouette_avg = silhouette_score(x_scaled, cluster_labels)
    print(f"Number of clusters: {n_clusters}, Silhouette Score: {silhouette_avg:.3f}")
    
    if silhouette_avg > best_score:
        best_score = silhouette_avg
        best_n_clusters = n_clusters

print(f"Best Silhouette Score: {best_score:.3f} with {best_n_clusters} clusters")


# %%
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

def find_best_linkage(x_scaled, n_clusters=4):
    best_davies = float('inf')
    best_score = -1
    best_linkage = None
    linkages = ['ward', 'complete', 'average', 'single']
    
    for linkage in linkages:
        clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        cluster_labels = clustering.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, cluster_labels)
        davies = davies_bouldin_score(x_scaled, cluster_labels)
        print(f"Linkage: {linkage}, Silhouette Score: {score:.3f}")
        print(f'Linkage: {linkage}, Davies Score: {davies}')
        if score > best_score and davies < best_davies:
            best_score = score
            best_linkage = linkage
            best_davies = davies
    
    print(f"\nBest Linkage: {best_linkage} with a Silhouette Score of {best_score:.3f} and a Davies Bouldin score of {best_davies}")
find_best_linkage(x_scaled, n_clusters=4)
h_davies = davies_bouldin_score(x_scaled, cluster_labels)

# %% [markdown]
# ### 3.2.4 Results and optimal hierachichal clustering algorithm
# 
# - After iterating through the different linkages, the **single** linkage produces the highest linkage matrix with a silhouette score of 0.41. 
# - Therefore, the optimal hierachichal cluster algorithm has the following parameters: 
#     - linkage = Single
#     - number of cluster = 4 
#     - Resulting Silhouette score = 0.409
#     - Resulting Davies Bouldin score = 0.561
# 

# %% [markdown]
# ## 3.3 Density-Based Spatial Clustering of Applications with Noise (DBSCAN) Algorithm
# 

# %% [markdown]
# ### 3.3.1 DBSCAN parameters to understand
# - when implementing DBACAN alg, the following parameters are crucial
#     - *eps*: maximum distance between clusters for datapoints to be considered neighbors. Not fixed, chosen based on each dataset
#     - *min_samples*: number of samples within a neighborhood for a point to be considered as a cluster

# %%
from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps = 2, min_samples = 2)
clusters = dbscan.fit_predict(x_scaled)

# %%
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
n_noise_ = list(clusters).count(-1)
print("Estimated number of clusters: %d" % n_clusters)
print("Estimated number of noise points %d" % n_noise_)


# %% [markdown]
# ### 3.3.1 Observations
# 
# - The initial number of eps had been set to 2 and the min_samples was set as 2, however this resulted in 36 cluster and 167 noise points which meant that the clustering was not very effective.
# - The silhouette score calculated below validates this inference

# %% [markdown]
# ### 3.3.2 Visualization of clusters
# 

# %%
plt.figure(figsize=(10, 6))
unique_labels = set(clusters)
for label in unique_labels:
    if label == 1:
        plt.plot(x_scaled[clusters == label, 0], x_scaled[clusters == label, 1], 'k+',  markersize=10)
    else:
         plt.plot(x_scaled[clusters == label, 0], x_scaled[clusters == label, 1], 'o', markerfacecolor=plt.cm.nipy_spectral(label / n_clusters), markersize=6)
plt.title('DBSCAN Clustering')
plt.show()

# %%
silhouette_avg = silhouette_score(x_scaled, clusters)
print(f"Silhouette Score: {silhouette_avg:.3f}")
DB_davies = davies_bouldin_score(x_scaled, clusters)
print(f'DBSCAN Davies Score: {DB_davies}')

# %% [markdown]
# ### 3.3.3 Adjustment of Parameters to find optimum
# - from the initial implementation of DBSCAN above, the resulting silhouette score is very poor, standing at **-0.045**
# - much like in the hierachichal clustering algorithm, the I iterate through different values of eps and Min_samples to find one that produces the most optimal silhouette score. 
# - The highest resulting silhouette score is chosen as the optimal values of eps and min_samples

# %%
eps_values = np.arange(1, 5)
min_samples_values = range(2, 10)
lowest_davies = float('inf')
best_sil_eps = None
best_sil_min_samples = None
highest_silhouette_score = -1
best_dav_eps = None
best_dav_min_samples = None

for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(x_scaled)
        if len(set(clusters)) - (1 if -1 in clusters else 0) > 1:
            score = silhouette_score(x_scaled, clusters)
            davies = davies_bouldin_score(x_scaled, clusters)
            if score > highest_silhouette_score:
                highest_silhouette_score = score
                best_sil_eps = eps
                best_sil_min_samples = min_samples
                best_clusters = clusters
            if davies < lowest_davies:
                best_dav_eps = eps
                best_dav_min_samples = min_samples
                lowest_davies = davies

print(f"Best silhouette eps: {best_sil_eps}, Best silhouette min_samples: {best_sil_min_samples}, Highest Silhouette Score: {highest_silhouette_score:.3f}")
print(f"Best davies eps: {best_dav_eps}, Best davies min_samples: {best_dav_min_samples}, Lowest Davies Score: {lowest_davies:.3f}")

# %% [markdown]
# ### 3.3.4 Final DBSCAN parameters and results
# - The final parameters chosen for the DBSCAN clustering are as follows:
#     - eps: 4
#     - min_samples: 5
#     - Best_silhouette score: 0.125
#     - Best-Bouldin score: 1.256
#     - Ultimately, in comparison to the hierachical clustering method defined above, DBSCAN produces the least silhouette score of 0.125 and the highest bouldin score of 1.256 making it lesser preffered to the hierachichal clustering model

# %% [markdown]
# # 4.0 Gaussian Mixture Model
# 

# %%
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(x_scaled)

labels = gmm.predict(x_scaled)

GMM_silhouette = silhouette_score(x_scaled, labels)
GMM_davies_score = davies_bouldin_score(x_scaled, labels)
print(f"GMM Silhouette Score: {score:.2f}")
print(f"GMM Davies score: {GMM_davies_score:.2f}")

print("Means of each component (cluster center in the scaled feature space):")
print(gmm.means_)

# %% [markdown]
# ### 4.1.1 GMM model interpretation
# - The features with the highest absolute mean values across the centroids are:
# 
#     - Feature 15 (Cp-3): 2.394438 in the first cluster.
#     - Feature 20 (restecg-1): -0.69663055 in the second cluster.
#     - Feature 20(restecg-1): 1.38531379 in the third clusters.

# %% [markdown]
# # 5.0 Model Evaluation and Selection Based on Metrics
# 
# - After the exhaustive exploration of the 4 clustering methods above, the following metrics were obtained for each model:
#     - K-means clustering model:
#         - silhouette score: **0.103**
#         - davies bouldin score: **2.83**
#     - Hierachichal clustering model:
#         - silhouette score: **0.562**
#         - davies bouldin score: **0.409**
#     - DBSCAN clustering model:
#         - silhouette score: **0.125**
#         - davies bouldin score: **1.256**
#     - Gausian mixture model:
#         - silhouette score: **0.10**
#         - davies bouldin score: **2.06**
# - With these results, it is evident that the **hierachichal clustering model** produces the highest silhouette score and the lowest davies bouldin score making it the most optimal clustering model for the heart disease dataset herein


