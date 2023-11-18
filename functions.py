import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA

from sklearn import set_config
set_config(transform_output="pandas")



def get_dataframe_heatmap(audio_features_df, scaler_transformer):

    # Create a scaler object
    scaler = scaler_transformer().set_output(transform="pandas")
    # Scale the audio_features_df DataFrame
    audio_features_df_temp = scaler.fit_transform(audio_features_df)

    # Making the DataFrame for a heatmap
    audio_features_df_distances = pd.DataFrame(pairwise_distances(audio_features_df_temp),
                                            index=audio_features_df.index,
                                            columns=audio_features_df.index)

    # Create a figure with a size of 12 inches by 8 inches
    plt.subplots(figsize=(12, 8))

    # Generate a heatmap of the Euclidean distances DataFrame
    sns.heatmap(audio_features_df_distances);

    return audio_features_df_temp, audio_features_df_distances



def get_dataframe_scaled(audio_features_df, scaler_transformer):

    # Create a scaler object
    scaler = scaler_transformer().set_output(transform="pandas") 

    # Scale the audio_features_df DataFrame
    audio_features_df_scaled = scaler.fit_transform(audio_features_df)

    return audio_features_df_scaled



def get_dataframe_quentiletransformed(audio_features_df):
    # Number of samples
    number_of_samples = audio_features_df.shape[0]

    # Create a QuantileTransformer object
    quantile_scaler = QuantileTransformer(n_quantiles = number_of_samples).set_output(transform="pandas")

    # Transform the audio_features_df DataFrame
    audio_features_df_quantile = quantile_scaler.fit_transform(audio_features_df)

    return audio_features_df_quantile



def get_kmeansdf_heatmap(audio_features_df, k):
    # Initialise the model
    my_kmeans_temp = KMeans(n_clusters= k, # you always choose the number of k here
                    n_init="auto",
                    random_state = 123)

    # Fit the model to the data
    my_kmeans_temp.fit(audio_features_df)

    # Obtain the cluster output
    clusters = my_kmeans_temp.labels_

    # Preparing a new Dataframe with clusters-column
    audio_features_df_clustered = audio_features_df.copy()

    # Attach the cluster output to prepared DataFrame
    audio_features_df_clustered["cluster"] = clusters

    # Find the coordinates of each centroid using the cluster_centers_ attribute
    centroids = my_kmeans_temp.cluster_centers_

    # Calculate the euclidean distance between the centroids
    centroid_distances_temp = pairwise_distances(centroids)

    # Plot distances on heatmap
    sns.heatmap(centroid_distances_temp,
                #annot=True,
                linewidths=1);

    return audio_features_df_clustered, centroid_distances_temp



def get_radar_chart(scaled_features_df):
    # State the label for each arm of the chart
    categories = ['danceability', 'energy', 'key', 'loudness',
        'mode', 'speechiness', 'acousticness', 'liveness', 'valence', 'tempo', 'duration_ms', 'time_signature']

    # Create an empty list to store the objects
    trace_objects = []

    # Iterate over the unique cluster numbers and add an object for each cluster to the list
    for cluster in sorted(scaled_features_df['cluster'].unique()):
        cluster_food_means = go.Scatterpolar(
            r=[scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'danceability'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'energy'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'key'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'loudness'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'mode'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'speechiness'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'acousticness'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'liveness'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'valence'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'tempo'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'duration_ms'].mean(),
                scaled_features_df.loc[scaled_features_df["cluster"] == cluster, 'time_signature'].mean()],
            theta=categories,
            fill='toself',
            name=f'Cluster {cluster}'
        )
        trace_objects.append(cluster_food_means)

    # Add the objects to the figure
    fig = go.Figure()
    fig.add_traces(trace_objects)

    # Add extras to the plot such as title
    # You'll always need `polar=dict(radialaxis=dict(visible=True,range=[0, 1]))` when creating a radar plot
    fig.update_layout(
        title_text = 'Radar chart of mean audio features by cluster',
        height = 600,
        width = 800,
        polar=dict(
            radialaxis=dict(
            visible=True,
            range=[0, 1]
            )),
        showlegend=True
    )

    # Show the initialised plot and the layers
    fig.show()



def get_inertia_elbow_method(scaled_features_df_inertia, maximum_k_inertia):
    # Decide on a random_state to use
    seed = 123

    # Set the maximum number of clusters to try
    max_k = maximum_k_inertia

    # Create an empty list to store the inertia scores
    inertia_list = []

    # Iterate over the range of cluster numbers
    for i in range(1, max_k):

        # Create a KMeans object with the specified number of clusters
        myKMeans = KMeans(n_clusters=i,
                        n_init="auto",
                        random_state = seed)

        # Fit the KMeans model to the scaled data
        myKMeans.fit(scaled_features_df_inertia)

        # Append the inertia score to the list
        inertia_list.append(myKMeans.inertia_)

    # Set the Seaborn theme to darkgrid
    sns.set_theme(style='darkgrid')

    (
    # Create a line plot of the inertia scores
    sns.relplot(y = inertia_list,
                x = range(1,max_k),
                kind = 'line',
                marker = 'o',
                height = 8,
                aspect = 2)
    # Set the title of the plot
    .set(title=f"Inertia score from 1 to {max_k} clusters")
    # Set the axis labels
    .set_axis_labels("Number of clusters", "Inertia score")
    )



def get_silhouette_score(scaled_features_df, maximum_k):
    # Decide on a random_state to use
    seed = 123

    # Set the maximum number of clusters to try
    max_k = maximum_k

    # Create an empty list to store the silhouette scores
    sil_scores = []


    for j in range(2, max_k):

        # Create a KMeans object with the specified number of clusters
        kmeans = KMeans(n_clusters=j,
                        n_init="auto",
                        random_state = seed)

        # Fit the KMeans model to the scaled data
        kmeans.fit(scaled_features_df)

        # Get the cluster labels
        labels = kmeans.labels_

        # Calculate the silhouette score
        score = silhouette_score(scaled_features_df, labels)

        # Append the silhouette score to the list
        sil_scores.append(score)

    sns.set_theme(style='darkgrid')

    (
    sns.relplot(y=sil_scores,
                x=range(2,max_k),
                kind='line',
                marker='o',
                height = 8,
                aspect=2)
    .set(title=f"Silhouette score from 2 to {max_k} clusters")
    .set_axis_labels("Number of clusters", "Silhouette score")
    );



def get_pca_elbow(scaled_features_df):
    # Initialise the PCA object
    pca = PCA()

    # Fit the PCA object to the data
    pca.fit(scaled_features_df)

    # Transform scaled_features_df based on the fit calculations
    pca_basic_df = pca.transform(scaled_features_df)

    pca_basic_df

    # Get the variance explained by each principal component
    explained_variance_array = pca.explained_variance_ratio_

    explained_variance_array

    # Create a Pandas DataFrame from the variance explained array
    explained_variance_array_df = pd.DataFrame(explained_variance_array, columns=["Variance explained"])

    # Add a column for the principal component index
    explained_variance_array_df["Principal component index"] = range(len(explained_variance_array))

    (
    # Create a bar chart with sns.catplot
    sns.relplot(
        kind='line',
        data=explained_variance_array_df,
        x="Principal component index",
        y="Variance explained",
        marker='o',
        aspect=1.3)
    # Set the title of the plot
    .set(title="Proportion of variance explained by each principal component")
    # Set the axis labels
    .set_axis_labels("Principal component number", "Proportion of variance")
    );



def get_pca_variance(scaled_features_df):

    # Initialise the PCA object
    pca = PCA()

    # Fit the PCA object to the data
    pca.fit(scaled_features_df)

    # Transform scaled_features_df based on the fit calculations
    pca_basic_df = pca.transform(scaled_features_df)

    pca_basic_df

    # Get the variance explained by each principal component
    explained_variance_array = pca.explained_variance_ratio_

    explained_variance_array


    (
    # Create a cumulative explained variance plot
    sns.relplot(
        kind="line",  # Create a line plot
        x=np.arange(len(explained_variance_array)),  # Set the x-axis to be the principal component index
        y=np.cumsum(explained_variance_array),  # Set the y-axis to be the cumulative explained variance
        marker="o",  # Use a circle marker for the data points
        aspect=1.4,  # Set the aspect ratio of the plot to be 1.4
    )
    # Set the title of the plot
    .set(title="Cumulative explained variance of principal components")
    # Set the axis labels
    .set_axis_labels("Principal component", "Cumulative explained variance")
    );

    # Add a horizontal red line at 0.95 on the y axis
    plt.axhline(y=0.9,
                color='red');


