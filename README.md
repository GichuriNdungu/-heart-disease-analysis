# Heart Disease Clustering Analysis

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Fetch Dataset](#fetch-dataset)
- [Data Exploration](#data-exploration)
- [Data Preprocessing](#data-preprocessing)
- [Clustering Techniques](#clustering-techniques)
  - [K-Means Clustering](#k-means-clustering)
  - [Hierarchical Clustering](#hierarchical-clustering)
  - [DBSCAN Clustering](#dbscan-clustering)
  - [Gaussian Mixture Model](#gaussian-mixture-model)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)

## Overview
This project aims to analyze the Heart Disease dataset from the UCI Machine Learning Repository. The goal is to identify patterns and clusters related to heart disease risk factors by employing various clustering techniques, including K-Means, Hierarchical Clustering, DBSCAN, and Gaussian Mixture Models.

## Dataset
The Heart Disease dataset from the UCI Machine Learning Repository (ID: 45) is utilized, featuring various diagnostic attributes related to heart disease.

## Dependencies
- Python 3.6+
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- ucimlrepo

## Installation
To set up the project environment, install the required dependencies using the following command:
```bash
pip install pandas matplotlib seaborn scikit-learn scipy ucimlrepo
```

## Fetch Dataset
The dataset is acquired through the `ucimlrepo` package and loaded into pandas dataframes, separating features (`x`) and targets (`y`).

## Data Exploration
- Histograms display the distribution of features and targets.
- Scatter plots illustrate the relationship between age and heart disease prevalence.

## Data Preprocessing
- Missing values are addressed by median replacement.
- Categorical features undergo one-hot encoding.
- Feature scaling is performed using `StandardScaler`.

## Clustering Techniques

### K-Means Clustering
- A K-Means model is fitted with `k=5` clusters.
- The Elbow method assesses the optimal cluster count.
- PCA visualizes the clusters.
- Silhouette and Davies-Bouldin scores are calculated for evaluation.

### Hierarchical Clustering
- A dendrogram visualizes the clustering process.
- The Agglomerative Clustering model is applied.
- Silhouette scores help optimize the cluster number.
- The best linkage method is determined.

### DBSCAN Clustering
- DBSCAN is implemented with initial parameters.
- Cluster visualization is provided.
- Parameters `eps` and `min_samples` are optimized.

### Gaussian Mixture Model
- The GMM predicts clusters.
- Evaluation metrics include silhouette and Davies-Bouldin scores.

## Model Evaluation
The clustering models are compared using silhouette and Davies-Bouldin scores to determine the most effective approach for the dataset.

## Results
Hierarchical Clustering emerged as the most optimal model, achieving the highest silhouette score (0.562) and the lowest Davies-Bouldin score (0.409).

## Conclusion
The analysis successfully applies various clustering techniques to the heart disease dataset, with Hierarchical Clustering providing the most insightful patterns related to heart disease risk factors.
