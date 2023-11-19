# Unsupervised Machine Learning
## Clustering over 5000 songs with Kmeans Method

## Project Objective
In this Project I want to cluster songs from a dataset according to their provided audio features, like loudness, energy, danceability etc. to create playlists which I will push to Spotify afterwards.
I will use different Scalers and Transformers, like the MinMax- or StandardScaler or the Power Transformer to normalize the audio features and compute the inertia and silhouette score to decide which scaler is a good choice.
To cluster the songs, I will be using the Kmeans algorithm which is a popular clustering method and is based on minimizing the distances between the data points and the mean value of the cluster which will become the centroids in the next iteration.
Additionally I will use the PCA (Principal Components Analysis) method to reduce the dimensionality of the dataset by focusing on the most informative components (over 90% of summed variance is retained).
I've created several functions in a separate file to help with the repeated analysis and clustering.

## Libraries and Dependicies
- pandas
- seaborn
- scikit-learn
- plotly
- matplotlib
- numpy
- spotipy
- pickle

## Presentation
[Google Slides](https://docs.google.com/presentation/d/1epneZ_r4a586vH3mTCswpuGTxYtuiJfnEgJ4Hsfw6Hs/edit#slide=id.p)
