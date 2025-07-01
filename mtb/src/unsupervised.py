"""

K-means Clustering
Partitions data into a fixed number of clusters
Try a variety of k values (number of clusters)
Visualization includes cluster distribution, feature importance, and temporal segments

DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
Identifies clusters of varying shapes and can detect outliers (noise)
Excellent for finding unusual events or anomalies
Doesn't require specifying the number of clusters in advance
Visualization highlights noise points and shows how clusters appear over time

t-SNE (t-Distributed Stochastic Neighbor Embedding)
Dimensionality reduction technique that preserves local structure
Good for visualizing high-dimensional sensor data in 2D or 3D
Shows relationships between data points and potential clusters
The visualization includes a time path through the embedded space to track the ride

HMM (Hidden Markov Model)
Models sequential data as a series of hidden states
Identifies different "states" or modes of riding over time
Shows state transitions and temporal patterns in ride
Particularly good for capturing the sequential nature of data

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from hmmlearn import hmm
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import plotly.express as px
import plotly.graph_objects as go
from util import df_info_to_file


def analyze_pca_components(
    df: pd.DataFrame,
    selected_cols: List[str] | None = None,
    title: str = "PCA Variance Analysis",
    standardize: bool = True,
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on selected columns and visualize variance explained to help determine
    the optimal number of components.

    Parameters:
        df (pd.DataFrame): Input DataFrame with data for PCA
        selected_cols (List[str]): Column names to use for PCA
        title (str): Title for the plot
        standardize (bool): Whether to standardize the data before PCA
        plot (bool): Whether to generate and display the plot
        verbose (bool): Whether to print information about the PCA results

    Returns:
        Tuple containing:
        - pca: The fitted PCA object
        - X: The input data matrix
        - explained_variance_ratio: Array of explained variance ratios
        - cumulative_variance: Array of cumulative explained variance
    """

    # Check if all selected columns are in the DataFrame
    if selected_cols:
        missing_cols = [col for col in selected_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"The following columns are not in the DataFrame: {missing_cols}"
            )

    # Extract selected features and handle missing values
    if selected_cols:
        X = df[selected_cols].dropna().values
    else:
        X = df.dropna().values
    n_samples, n_features = X.shape

    if verbose:
        print(f"PCA analysis using {n_features} features and {n_samples} samples")

    # Standardize the data if requested
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Perform PCA with all possible components
    n_components = min(n_samples, n_features)
    pca = PCA(n_components=n_components)
    pca.fit(X)

    # Get variance explained by each component
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)

    if verbose:
        print(f"Total explained variance: {cumulative_variance[-1]:.4f}")
        # Print variance explained by first few components
        for i in range(min(5, n_components)):
            print(
                f"Component {i+1}: {explained_variance_ratio[i]:.4f} "
                f"(Cumulative: {cumulative_variance[i]:.4f})"
            )

    # Create plots if requested
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Individual explained variance
        ax1.bar(range(1, n_components + 1), explained_variance_ratio, alpha=0.7)
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")
        ax1.set_title("Variance Explained by Each Component")
        ax1.set_xticks(
            range(1, min(n_components + 1, 21), 2)
        )  # Label every other component up to 20

        # Cumulative explained variance
        ax2.plot(
            range(1, n_components + 1), cumulative_variance, marker="o", linestyle="-"
        )
        ax2.axhline(y=0.95, color="r", linestyle="--", label="95% Variance")
        ax2.axhline(y=0.99, color="g", linestyle="--", label="99% Variance")

        # Find components needed for 95% and 99% variance
        components_95 = np.argmax(cumulative_variance >= 0.95) + 1
        components_99 = np.argmax(cumulative_variance >= 0.99) + 1

        if components_95 <= n_components:
            ax2.axvline(
                x=components_95,
                color="r",
                linestyle=":",
                label=f"{components_95} components for 95%",
            )

        if components_99 <= n_components:
            ax2.axvline(
                x=components_99,
                color="g",
                linestyle=":",
                label=f"{components_99} components for 99%",
            )

        ax2.set_xlabel("Number of Components")
        ax2.set_ylabel("Cumulative Explained Variance")
        ax2.set_title("Cumulative Variance Explained")
        ax2.set_xticks(
            range(1, min(n_components + 1, 21), 2)
        )  # Label every other component up to 20
        ax2.legend(loc="lower right")

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

        if verbose:
            print(f"Components needed for 95% variance: {components_95}")
            print(f"Components needed for 99% variance: {components_99}")

    return pca, X, explained_variance_ratio, cumulative_variance


def perform_final_pca(df, selected_cols, n_components, standardize=True):
    """
    Perform the final PCA with the optimal number of components determined from analysis.
    """

    # Extract selected features and handle missing values
    X = df[selected_cols].dropna().values

    # Standardize the data if requested
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Perform PCA with the selected number of components
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(X)

    # Create a DataFrame with the transformed data
    pca_df = pd.DataFrame(
        transformed_data,
        columns=[f"PC{i+1}" for i in range(n_components)],
        index=df[selected_cols].dropna().index,
    )

    return pca, pca_df


def plot_pca_components(
    pca_df: pd.DataFrame,
    original_df: pd.DataFrame = None,
    color_col: str = None,
    title: str = "Principal Component Analysis",
    alpha: float = 0.7,
    figsize: tuple = (12, 8),
    add_points: bool = True,
    add_lines: bool = True,
    max_points: int = 1000,  # Limit points for performance
    point_size: float = 20,
) -> plt.Figure:
    """
    (NOT VERY USEFUL AS IS!)

    Plot the first two principal components (PC1 vs PC2) as lines and/or scatter points.

    Parameters:
        pca_df (pd.DataFrame): DataFrame containing principal components (PC1, PC2, etc.)
        original_df (pd.DataFrame, optional): Original DataFrame with potential color column
        color_col (str, optional): Column in original_df to use for coloring points
        title (str): Title for the plot
        alpha (float): Transparency for the points
        figsize (tuple): Figure size
        add_points (bool): Whether to add scatter points
        add_lines (bool): Whether to add a line plot
        max_points (int): Maximum number of points to plot (for performance)
        point_size (float): Size of scatter points

    Returns:
        fig: The matplotlib Figure object

    Can also plot composition of PCs based on original features:

        plt_comp = 0
        plt.bar(pca_cols, pca.components_[0], alpha=0.7)
        plt.xticks(rotation=45)
        plt.ylabel('Component 1 Coefficients')
        plt.title(f'PCA Component {plt_comp + 1} Coefficients')
        plt.tight_layout()
        plt.show()


    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # If we have too many points, sample a subset for display
    plot_df = pca_df
    if len(pca_df) > max_points:
        # Sample points, but ensure we preserve the time order
        step = len(pca_df) // max_points
        plot_df = pca_df.iloc[::step].copy()

    # Plot settings
    if "PC1" not in plot_df.columns or "PC2" not in plot_df.columns:
        raise ValueError("DataFrame must contain 'PC1' and 'PC2' columns")

    # Determine color values if color_col provided
    colors = None
    if color_col and original_df is not None:
        if color_col in original_df.columns:
            # Match indices between pca_df and original_df
            matched_orig_df = original_df.loc[plot_df.index]
            colors = matched_orig_df[color_col]
        else:
            print(
                f"Warning: Color column '{color_col}' not found in original DataFrame"
            )

    # Add line plot
    if add_lines:
        ax.plot(
            plot_df["PC1"], plot_df["PC2"], "-", alpha=0.7, linewidth=1.5, color="navy"
        )

    # Add scatter plot
    if add_points:
        scatter = ax.scatter(
            plot_df["PC1"],
            plot_df["PC2"],
            c=colors,
            alpha=alpha,
            s=point_size,
            cmap="viridis",
        )

        # Add colorbar if we have colors
        if colors is not None:
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label(color_col)

    # Add labels and title
    ax.set_xlabel("Principal Component 1", fontsize=12)
    ax.set_ylabel("Principal Component 2", fontsize=12)
    ax.set_title(title, fontsize=16)

    # Add grid and equal aspect ratio for better visualization
    ax.grid(alpha=0.3)

    # Add annotations for first and last point if we have enough points
    if len(plot_df) >= 2:
        first_point = plot_df.iloc[0]
        last_point = plot_df.iloc[-1]

        ax.annotate(
            "Start",
            xy=(first_point["PC1"], first_point["PC2"]),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

        ax.annotate(
            "End",
            xy=(last_point["PC1"], last_point["PC2"]),
            xytext=(10, 10),
            textcoords="offset points",
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

    plt.tight_layout()
    return fig


def apply_kmeans(
    df: pd.DataFrame,
    features: List[str],
    n_clusters: int = 5,
    standardize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, KMeans]:
    """
    Apply KMeans clustering to the selected features.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns to use for clustering
        n_clusters (int): Number of clusters for KMeans
        standardize (bool): Whether to standardize the data before clustering
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (cluster labels, fitted KMeans model)
    """
    # Extract features and handle missing values
    X = df[features].dropna().values

    # Standardize if required
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X)

    return labels, kmeans


def visualize_kmeans(
    df: pd.DataFrame,
    features: List[str],
    labels: np.ndarray,
    kmeans_model: KMeans,
    time_col: str = "elapsed_seconds",
    dims: int = 2,
    plot_centroids: bool = True,
    plot_time_series: bool = True,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Visualize KMeans clustering results.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns used for clustering
        labels (np.ndarray): Cluster labels from KMeans
        kmeans_model (KMeans): Fitted KMeans model
        time_col (str): Column name for time information
        dims (int): Dimensions for visualization (2 or 3)
        plot_centroids (bool): Whether to plot cluster centroids
        plot_time_series (bool): Whether to plot clusters over time
        figsize (Tuple[int, int]): Figure size
    """
    # Create a copy of df with just the features used for clustering
    X_df = df[features].dropna().copy()

    # Add labels and time information
    X_df["Cluster"] = labels
    X_df["Time"] = df.loc[X_df.index, time_col]

    # Create matplotlib figure
    fig = plt.figure(figsize=figsize)

    # For 2D visualization
    if dims == 2:
        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(
                StandardScaler().fit_transform(X_df[features].values)
            )
            X_df["PC1"] = X_pca[:, 0]
            X_df["PC2"] = X_pca[:, 1]
            plot_features = ["PC1", "PC2"]
            title_prefix = "PCA projection of "
        else:
            plot_features = features
            title_prefix = ""

        # Plot scatter of points colored by cluster
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(
            X_df[plot_features[0]],
            X_df[plot_features[1]],
            c=X_df["Cluster"],
            cmap="viridis",
            alpha=0.7,
            s=30,
        )

        # Add centroids if requested
        if plot_centroids:
            if len(features) > 2:
                # Transform centroids to PCA space
                centroids_pca = pca.transform(
                    StandardScaler().fit_transform(kmeans_model.cluster_centers_)
                )
                ax1.scatter(
                    centroids_pca[:, 0],
                    centroids_pca[:, 1],
                    marker="*",
                    s=300,
                    c="red",
                    edgecolor="k",
                    label="Centroids",
                )
            else:
                ax1.scatter(
                    kmeans_model.cluster_centers_[:, 0],
                    kmeans_model.cluster_centers_[:, 1],
                    marker="*",
                    s=300,
                    c="red",
                    edgecolor="k",
                    label="Centroids",
                )

        ax1.set_title(f"{title_prefix}KMeans Clustering")
        ax1.set_xlabel(plot_features[0])
        ax1.set_ylabel(plot_features[1])
        ax1.legend()

        # Add colorbar for cluster labels
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Cluster")

    else:  # 3D visualization
        # If more than 3 features, use PCA for visualization
        if len(features) > 3:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(
                StandardScaler().fit_transform(X_df[features].values)
            )
            X_df["PC1"] = X_pca[:, 0]
            X_df["PC2"] = X_pca[:, 1]
            X_df["PC3"] = X_pca[:, 2]
            plot_features = ["PC1", "PC2", "PC3"]
            title_prefix = "PCA projection of "
        else:
            plot_features = features[:3]
            title_prefix = ""

        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        scatter = ax1.scatter(
            X_df[plot_features[0]],
            X_df[plot_features[1]],
            X_df[plot_features[2]],
            c=X_df["Cluster"],
            cmap="viridis",
            alpha=0.7,
            s=30,
        )

        # Add centroids if requested
        if plot_centroids:
            if len(features) > 3:
                # Transform centroids to PCA space
                centroids_pca = pca.transform(
                    StandardScaler().fit_transform(kmeans_model.cluster_centers_)
                )
                ax1.scatter(
                    centroids_pca[:, 0],
                    centroids_pca[:, 1],
                    centroids_pca[:, 2],
                    marker="*",
                    s=300,
                    c="red",
                    edgecolor="k",
                    label="Centroids",
                )
            else:
                ax1.scatter(
                    kmeans_model.cluster_centers_[:, 0],
                    kmeans_model.cluster_centers_[:, 1],
                    kmeans_model.cluster_centers_[:, 2],
                    marker="*",
                    s=300,
                    c="red",
                    edgecolor="k",
                    label="Centroids",
                )

        ax1.set_title(f"{title_prefix}KMeans Clustering")
        ax1.set_xlabel(plot_features[0])
        ax1.set_ylabel(plot_features[1])
        ax1.set_zlabel(plot_features[2])
        ax1.legend()

    # Plot features by cluster (boxplot)
    ax2 = fig.add_subplot(2, 2, 2)
    # Melt the dataframe to long format for boxplot
    melted = pd.melt(
        X_df[features + ["Cluster"]],
        id_vars=["Cluster"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )
    # Create boxplot
    sns.boxplot(x="Feature", y="Value", hue="Cluster", data=melted, ax=ax2)
    ax2.set_title("Feature Distribution by Cluster")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # Plot cluster distribution
    ax3 = fig.add_subplot(2, 2, 3)
    cluster_counts = X_df["Cluster"].value_counts().sort_index()
    ax3.bar(cluster_counts.index, cluster_counts.values)
    ax3.set_title("Cluster Sizes")
    ax3.set_xlabel("Cluster")
    ax3.set_ylabel("Count")

    # Plot time series colored by cluster
    if plot_time_series:
        ax4 = fig.add_subplot(2, 2, 4)
        # Sort by time to ensure proper time-series visualization
        X_df_sorted = X_df.sort_values("Time")
        scatter = ax4.scatter(
            X_df_sorted["Time"],
            np.zeros(len(X_df_sorted)),  # y-values are arbitrary for visualization
            c=X_df_sorted["Cluster"],
            cmap="viridis",
            alpha=0.7,
            s=20,
        )
        ax4.set_title("Cluster Assignment over Time")
        ax4.set_xlabel(time_col)
        ax4.set_yticks([])  # Hide y-axis ticks

        # Add colorbar
        plt.colorbar(scatter, ax=ax4, label="Cluster")

    plt.tight_layout()
    plt.show()


def apply_dbscan(
    df: pd.DataFrame,
    features: List[str],
    eps: float = 0.5,
    min_samples: int = 5,
    standardize: bool = True,
) -> Tuple[np.ndarray, DBSCAN]:
    """
    Apply DBSCAN clustering to the selected features.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns to use for clustering
        eps (float): Maximum distance between samples in the same neighborhood
        min_samples (int): Minimum number of samples in a neighborhood to form a core point
        standardize (bool): Whether to standardize the data before clustering

    Returns:
        Tuple of (cluster labels, fitted DBSCAN model)
    """
    # Extract features and handle missing values
    X = df[features].dropna().values

    # Standardize if required
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X)

    return labels, dbscan


def visualize_dbscan(
    df: pd.DataFrame,
    features: List[str],
    labels: np.ndarray,
    dbscan_model: DBSCAN,
    time_col: str = "elapsed_seconds",
    dims: int = 2,
    plot_time_series: bool = True,
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Visualize DBSCAN clustering results.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns used for clustering
        labels (np.ndarray): Cluster labels from DBSCAN
        dbscan_model (DBSCAN): Fitted DBSCAN model
        time_col (str): Column name for time information
        dims (int): Dimensions for visualization (2 or 3)
        plot_time_series (bool): Whether to plot clusters over time
        figsize (Tuple[int, int]): Figure size
    """
    # Create a copy of df with just the features used for clustering
    X_df = df[features].dropna().copy()

    # Add labels and time information
    X_df["Cluster"] = labels
    X_df["Time"] = df.loc[X_df.index, time_col]

    # Count the number of clusters (excluding noise points with label -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    # Create a custom colormap that makes noise points (-1) gray
    colors = plt.cm.viridis(np.linspace(0, 1, n_clusters))
    # Add gray for noise
    cmap = mcolors.ListedColormap(np.vstack([np.array([0.7, 0.7, 0.7, 1.0]), colors]))

    # Create matplotlib figure
    fig = plt.figure(figsize=figsize)

    # For 2D visualization
    if dims == 2:
        # If more than 2 features, use PCA for visualization
        if len(features) > 2:
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(
                StandardScaler().fit_transform(X_df[features].values)
            )
            X_df["PC1"] = X_pca[:, 0]
            X_df["PC2"] = X_pca[:, 1]
            plot_features = ["PC1", "PC2"]
            title_prefix = "PCA projection of "
        else:
            plot_features = features
            title_prefix = ""

        # Plot scatter of points colored by cluster
        ax1 = fig.add_subplot(2, 2, 1)
        scatter = ax1.scatter(
            X_df[plot_features[0]],
            X_df[plot_features[1]],
            c=X_df["Cluster"],
            cmap=cmap,
            alpha=0.7,
            s=30,
        )

        ax1.set_title(
            f"{title_prefix}DBSCAN Clustering: {n_clusters} clusters, {n_noise} noise points"
        )
        ax1.set_xlabel(plot_features[0])
        ax1.set_ylabel(plot_features[1])

        # Add colorbar for cluster labels
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label("Cluster")

    else:  # 3D visualization
        # If more than 3 features, use PCA for visualization
        if len(features) > 3:
            pca = PCA(n_components=3)
            X_pca = pca.fit_transform(
                StandardScaler().fit_transform(X_df[features].values)
            )
            X_df["PC1"] = X_pca[:, 0]
            X_df["PC2"] = X_pca[:, 1]
            X_df["PC3"] = X_pca[:, 2]
            plot_features = ["PC1", "PC2", "PC3"]
            title_prefix = "PCA projection of "
        else:
            plot_features = features[:3]
            title_prefix = ""

        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        scatter = ax1.scatter(
            X_df[plot_features[0]],
            X_df[plot_features[1]],
            X_df[plot_features[2]],
            c=X_df["Cluster"],
            cmap=cmap,
            alpha=0.7,
            s=30,
        )

        ax1.set_title(
            f"{title_prefix}DBSCAN Clustering: {n_clusters} clusters, {n_noise} noise points"
        )
        ax1.set_xlabel(plot_features[0])
        ax1.set_ylabel(plot_features[1])
        ax1.set_zlabel(plot_features[2])

    # Plot features by cluster (boxplot)
    ax2 = fig.add_subplot(2, 2, 2)
    # Melt the dataframe to long format for boxplot
    melted = pd.melt(
        X_df[features + ["Cluster"]],
        id_vars=["Cluster"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )
    # Create boxplot
    sns.boxplot(x="Feature", y="Value", hue="Cluster", data=melted, ax=ax2)
    ax2.set_title("Feature Distribution by Cluster")
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)

    # Plot cluster distribution
    ax3 = fig.add_subplot(2, 2, 3)
    cluster_counts = X_df["Cluster"].value_counts().sort_index()
    ax3.bar(cluster_counts.index, cluster_counts.values)
    ax3.set_title("Cluster Sizes")
    ax3.set_xlabel("Cluster (-1 = Noise)")
    ax3.set_ylabel("Count")

    # Plot time series colored by cluster
    if plot_time_series:
        ax4 = fig.add_subplot(2, 2, 4)
        # Sort by time to ensure proper time-series visualization
        X_df_sorted = X_df.sort_values("Time")
        scatter = ax4.scatter(
            X_df_sorted["Time"],
            np.zeros(len(X_df_sorted)),  # y-values are arbitrary for visualization
            c=X_df_sorted["Cluster"],
            cmap=cmap,
            alpha=0.7,
            s=20,
        )
        ax4.set_title("Cluster Assignment over Time")
        ax4.set_xlabel(time_col)
        ax4.set_yticks([])  # Hide y-axis ticks

        # Add colorbar
        plt.colorbar(scatter, ax=ax4, label="Cluster (-1 = Noise)")

    plt.tight_layout()
    plt.show()


def apply_tsne(
    df: pd.DataFrame,
    features: List[str],
    n_components: int = 2,
    perplexity: float = 30.0,
    learning_rate: float = 200.0,
    n_iter: int = 1000,
    standardize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, TSNE]:
    """
    Apply t-SNE dimensionality reduction to the selected features.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns to use for t-SNE
        n_components (int): Number of dimensions in the embedded space
        perplexity (float): Perplexity parameter for t-SNE
        learning_rate (float): Learning rate for t-SNE
        n_iter (int): Maximum number of iterations
        standardize (bool): Whether to standardize the data
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (embedded data array, fitted t-SNE model)
    """
    # Extract features and handle missing values
    X = df[features].dropna().values

    # Standardize if required
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        random_state=random_state,
    )
    X_embedded = tsne.fit_transform(X)

    return X_embedded, tsne


def visualize_tsne(
    df: pd.DataFrame,
    features: List[str],
    tsne_result: np.ndarray,
    color_col: Optional[str] = None,
    time_col: str = "elapsed_seconds",
    plot_time_path: bool = True,
    figsize: Tuple[int, int] = (14, 10),
) -> None:
    """
    Visualize t-SNE results.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns used for t-SNE
        tsne_result (np.ndarray): Result from t-SNE embedding
        color_col (str, optional): Column name to color points by
        time_col (str): Column name for time information
        plot_time_path (bool): Whether to plot the time path through t-SNE space
        figsize (Tuple[int, int]): Figure size
    """
    # Create a copy of df with just the features used for t-SNE
    X_df = df[features].dropna().copy()

    # Add time information
    X_df["Time"] = df.loc[X_df.index, time_col]

    # Add t-SNE dimensions
    if tsne_result.shape[1] >= 2:
        X_df["TSNE1"] = tsne_result[:, 0]
        X_df["TSNE2"] = tsne_result[:, 1]

    if tsne_result.shape[1] >= 3:
        X_df["TSNE3"] = tsne_result[:, 2]

    # Add color column if provided
    if color_col and color_col in df.columns:
        X_df["Color"] = df.loc[X_df.index, color_col]

    # Create matplotlib figure
    fig = plt.figure(figsize=figsize)

    # Plot the t-SNE results
    if tsne_result.shape[1] == 2:
        ax1 = fig.add_subplot(2, 2, 1)

        # Color by time if no color column specified
        if color_col is None or color_col not in df.columns:
            scatter = ax1.scatter(
                X_df["TSNE1"],
                X_df["TSNE2"],
                c=X_df["Time"],
                cmap="viridis",
                alpha=0.7,
                s=30,
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label("Time")
        else:
            scatter = ax1.scatter(
                X_df["TSNE1"],
                X_df["TSNE2"],
                c=X_df["Color"],
                cmap="viridis",
                alpha=0.7,
                s=30,
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label(color_col)

        ax1.set_title("t-SNE Embedding")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")

        # Plot the time path through the t-SNE space
        if plot_time_path:
            ax2 = fig.add_subplot(2, 2, 2)

            # Sort by time
            X_df_sorted = X_df.sort_values("Time")

            # Plot points with alpha to see density
            ax2.scatter(
                X_df_sorted["TSNE1"], X_df_sorted["TSNE2"], alpha=0.2, s=10, c="gray"
            )

            # Plot the path
            ax2.plot(
                X_df_sorted["TSNE1"],
                X_df_sorted["TSNE2"],
                "b-",
                alpha=0.7,
                linewidth=1.5,
            )

            # Mark the start and end
            ax2.scatter(
                X_df_sorted["TSNE1"].iloc[0],
                X_df_sorted["TSNE2"].iloc[0],
                color="green",
                s=100,
                label="Start",
            )
            ax2.scatter(
                X_df_sorted["TSNE1"].iloc[-1],
                X_df_sorted["TSNE2"].iloc[-1],
                color="red",
                s=100,
                label="End",
            )

            ax2.set_title("Time Path through t-SNE Space")
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")
            ax2.legend()

    elif tsne_result.shape[1] == 3:
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")

        # Color by time if no color column specified
        if color_col is None or color_col not in df.columns:
            scatter = ax1.scatter(
                X_df["TSNE1"],
                X_df["TSNE2"],
                X_df["TSNE3"],
                c=X_df["Time"],
                cmap="viridis",
                alpha=0.7,
                s=30,
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label("Time")
        else:
            scatter = ax1.scatter(
                X_df["TSNE1"],
                X_df["TSNE2"],
                X_df["TSNE3"],
                c=X_df["Color"],
                cmap="viridis",
                alpha=0.7,
                s=30,
            )
            cbar = plt.colorbar(scatter, ax=ax1)
            cbar.set_label(color_col)

        ax1.set_title("3D t-SNE Embedding")
        ax1.set_xlabel("t-SNE Component 1")
        ax1.set_ylabel("t-SNE Component 2")
        ax1.set_zlabel("t-SNE Component 3")

        # Plot the time path through the t-SNE space
        if plot_time_path:
            ax2 = fig.add_subplot(2, 2, 2, projection="3d")

            # Sort by time
            X_df_sorted = X_df.sort_values("Time")

            # Plot points with alpha to see density
            ax2.scatter(
                X_df_sorted["TSNE1"],
                X_df_sorted["TSNE2"],
                X_df_sorted["TSNE3"],
                alpha=0.2,
                s=10,
                c="gray",
            )

            # Plot the path
            ax2.plot(
                X_df_sorted["TSNE1"],
                X_df_sorted["TSNE2"],
                X_df_sorted["TSNE3"],
                "b-",
                alpha=0.7,
                linewidth=1.5,
            )

            # Mark the start and end
            ax2.scatter(
                X_df_sorted["TSNE1"].iloc[0],
                X_df_sorted["TSNE2"].iloc[0],
                X_df_sorted["TSNE3"].iloc[0],
                color="green",
                s=100,
                label="Start",
            )
            ax2.scatter(
                X_df_sorted["TSNE1"].iloc[-1],
                X_df_sorted["TSNE2"].iloc[-1],
                X_df_sorted["TSNE3"].iloc[-1],
                color="red",
                s=100,
                label="End",
            )

            ax2.set_title("Time Path through 3D t-SNE Space")
            ax2.set_xlabel("t-SNE Component 1")
            ax2.set_ylabel("t-SNE Component 2")
            ax2.set_zlabel("t-SNE Component 3")
            ax2.legend()

    # Plot feature distribution
    ax3 = fig.add_subplot(2, 2, 3)
    # Melt the dataframe to long format for boxplot
    melted = pd.melt(
        X_df[features], value_vars=features, var_name="Feature", value_name="Value"
    )
    # Create boxplot
    sns.boxplot(x="Feature", y="Value", data=melted, ax=ax3)
    ax3.set_title("Feature Distribution")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

    # Plot time series of a few selected features
    ax4 = fig.add_subplot(2, 2, 4)
    # Select a subset of features for clarity
    plot_features = features[: min(3, len(features))]
    for feature in plot_features:
        ax4.plot(X_df["Time"], X_df[feature], label=feature)
    ax4.set_title("Feature Time Series")
    ax4.set_xlabel(time_col)
    ax4.set_ylabel("Value")
    ax4.legend()

    plt.tight_layout()
    plt.show()


def apply_hmm(
    df: pd.DataFrame,
    features: List[str],
    n_components: int = 5,
    covariance_type: str = "full",
    n_iter: int = 100,
    standardize: bool = True,
    random_state: int = 42,
) -> Tuple[np.ndarray, hmm.GaussianHMM]:
    """
    Apply Hidden Markov Model to the selected features.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns to use for HMM
        n_components (int): Number of states in the HMM
        covariance_type (str): Type of covariance 'full', 'tied', 'diag', 'spherical'
        n_iter (int): Maximum number of iterations
        standardize (bool): Whether to standardize the data
        random_state (int): Random seed for reproducibility

    Returns:
        Tuple of (predicted states, fitted HMM model)
    """
    # Extract features and handle missing values
    X = df[features].dropna().values

    # Standardize if required
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # Apply HMM
    model = hmm.GaussianHMM(
        n_components=n_components,
        covariance_type=covariance_type,
        n_iter=n_iter,
        random_state=random_state,
    )

    # Convert data to float64
    X = X.astype(np.float64)

    # Fit model and predict states
    model.fit(X)
    states = model.predict(X)

    return states, model


def visualize_hmm(
    df: pd.DataFrame,
    features: List[str],
    states: np.ndarray,
    hmm_model: hmm.GaussianHMM,
    time_col: str = "elapsed_seconds",
    figsize: Tuple[int, int] = (16, 12),
) -> None:
    """
    Visualize HMM states and transitions.

    Parameters:
        df (pd.DataFrame): Input DataFrame
        features (List[str]): List of feature columns used for HMM
        states (np.ndarray): Predicted states from HMM
        hmm_model (hmm.GaussianHMM): Fitted HMM model
        time_col (str): Column name for time information
        figsize (Tuple[int, int]): Figure size
    """
    # Create a copy of df with just the features used for HMM
    X_df = df[features].dropna().copy()

    # Add states and time information
    X_df["State"] = states
    X_df["Time"] = df.loc[X_df.index, time_col]

    # Create matplotlib figure
    fig = plt.figure(figsize=figsize)

    # Plot states over time
    ax1 = fig.add_subplot(2, 2, 1)
    # Sort by time to ensure proper time-series visualization
    X_df_sorted = X_df.sort_values("Time")
    scatter = ax1.scatter(
        X_df_sorted["Time"],
        np.zeros(len(X_df_sorted)),  # y-values are arbitrary for visualization
        c=X_df_sorted["State"],
        cmap="viridis",
        alpha=0.7,
        s=20,
    )
    ax1.set_title("HMM States over Time")
    ax1.set_xlabel(time_col)
    ax1.set_yticks([])  # Hide y-axis ticks

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label("State")

    # Plot state distribution
    ax2 = fig.add_subplot(2, 2, 2)
    state_counts = X_df["State"].value_counts().sort_index()
    ax2.bar(state_counts.index, state_counts.values)
    ax2.set_title("State Distribution")
    ax2.set_xlabel("State")
    ax2.set_ylabel("Count")

    # Plot features by state (boxplot)
    ax3 = fig.add_subplot(2, 2, 3)
    # Melt the dataframe to long format for boxplot
    melted = pd.melt(
        X_df[features + ["State"]],
        id_vars=["State"],
        value_vars=features,
        var_name="Feature",
        value_name="Value",
    )
    # Create boxplot
    sns.boxplot(x="Feature", y="Value", hue="State", data=melted, ax=ax3)
    ax3.set_title("Feature Distribution by State")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)

    # Plot transition matrix heatmap
    ax4 = fig.add_subplot(2, 2, 4)
    sns.heatmap(hmm_model.transmat_, annot=True, fmt=".2f", cmap="Blues", ax=ax4)
    ax4.set_title("Transition Matrix")
    ax4.set_xlabel("To State")
    ax4.set_ylabel("From State")

    plt.tight_layout()
    plt.show()


def plot_variable_by_cluster(
    df: pd.DataFrame,
    var_col: str,
    cluster_col: str = "kmeans5",
    time_col: str = None,  # Uses DatetimeIndex by default
    title: str = None,
    marker_size: int = 6,
    height: int = 600,
    width: int = 1000,
    template: str = "plotly_white",
    color_discrete_sequence: list = None,
    opacity: float = 0.8,
    show_legend: bool = True,
) -> None:
    """
    Create a Plotly scatter plot for a single variable with points colored by cluster labels.
    Uses DataFrame's DatetimeIndex by default for the x-axis.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data
        var_col (str): Name of the variable column to plot
        cluster_col (str): Name of the cluster label column
        time_col (str, optional): Name of the time column. If None, uses DataFrame's index
        title (str): Plot title (if None, auto-generated)
        marker_size (int): Size of markers
        height (int): Plot height
        width (int): Plot width
        template (str): Plotly template
        color_discrete_sequence (list): List of colors for different clusters
        opacity (float): Opacity of markers
        show_legend (bool): Whether to show the legend
    """
    import plotly.express as px

    # Ensure the dataframe has the required columns
    if var_col not in df.columns:
        raise ValueError(f"Variable column '{var_col}' not found in DataFrame")
    if cluster_col not in df.columns:
        raise ValueError(f"Cluster column '{cluster_col}' not found in DataFrame")

    # Create a copy of the dataframe to avoid modifying the original
    df_plot = df.copy()

    # Check if we need to use DatetimeIndex or a time column
    using_index = False
    if time_col is None:
        # Check if we have a DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError(
                "DataFrame must have a DatetimeIndex when time_col is None"
            )

        # Reset index to make it a column for Plotly
        df_plot = df_plot.reset_index()
        time_col = df_plot.columns[0]  # First column after reset_index
        using_index = True
    elif time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame")

    # Sort dataframe by time
    df_sorted = df_plot.sort_values(by=time_col)

    # Auto-generate title if not provided
    if title is None:
        title = f"{var_col} Over Time Colored by {cluster_col}"

    # Create the scatter plot (no lines, just markers)
    fig = px.scatter(
        df_sorted,
        x=time_col,
        y=var_col,
        color=cluster_col,
        hover_data=[cluster_col],
        title=title,
        height=height,
        width=width,
        template=template,
        color_discrete_sequence=color_discrete_sequence,
        opacity=opacity,
        size_max=marker_size,
    )

    # Update marker size
    fig.update_traces(marker=dict(size=marker_size))

    # Update layout
    x_axis_title = "Time" if using_index else time_col
    fig.update_layout(
        xaxis_title=x_axis_title,
        yaxis_title=var_col,
        legend_title=cluster_col,
        showlegend=show_legend,
        hovermode="closest",
    )

    # Show the plot
    fig.show()


def identify_jumps(
    df: pd.DataFrame,
    accel_col: str = "az_uc_e",
    threshold: float = 1.5,
    new_col: str = "jump",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Identify potential jumps in motion data by detecting when vertical
    acceleration is close to zero (free fall).

    Parameters:
        df (pd.DataFrame): Input DataFrame containing acceleration data
        accel_col (str): Column name for vertical acceleration (typically 'az_e')
        threshold (float): Threshold value around zero to identify as a jump
                          (smaller values = more precise jump detection)
        new_col (str): Name of the new column to create
        inplace (bool): If True, modify the DataFrame in place. Otherwise, return a new DataFrame.

    Returns:
        pd.DataFrame: DataFrame with the new 'jump' column (1 for jump, 0 for no jump)
    """
    # Check if acceleration column exists
    if accel_col not in df.columns:
        raise ValueError(f"Acceleration column '{accel_col}' not found in DataFrame")

    # Create a copy if not modifying in place
    if not inplace:
        df = df.copy()

    # Create the jump column
    # 1 where absolute value of acceleration is below threshold (near-zero = free fall)
    # 0 elsewhere
    df[new_col] = (df[accel_col].abs() < threshold).astype(int)

    return df


features = [
    "ar",
    "gr",
    "ar_uc",
    "ax_e",
    "ay_e",
    "az_e",
    "ax_uc_e",
    "ay_uc_e",
    "az_uc_e",
    "gx_e",
    "gy_e",
    "gz_e",
    "ax_t",
    "ay_t",
    "gx_t",
    "gy_t",
    "ax_uc_t",
    "ay_uc_t",
    "ar_lead1",
    "gr_lead1",
    "ar_uc_lead1",
    "az_e_lead1",
    "az_uc_e_lead1",
    "gz_e_lead1",
    "ax_t_lead1",
    "ay_t_lead1",
    "ax_uc_t_lead1",
    "ay_uc_t_lead1",
    "gx_t_lead1",
    "gy_t_lead1",
    "ar_lead2",
    "gr_lead2",
    "ar_uc_lead2",
    "az_e_lead2",
    "az_uc_e_lead2",
    "gz_e_lead2",
    "ax_t_lead2",
    "ay_t_lead2",
    "ax_uc_t_lead2",
    "ay_uc_t_lead2",
    "gx_t_lead2",
    "gy_t_lead2",
    "ar_lead3",
    "gr_lead3",
    "ar_uc_lead3",
    "az_e_lead3",
    "az_uc_e_lead3",
    "gz_e_lead3",
    "ax_t_lead3",
    "ay_t_lead3",
    "ax_uc_t_lead3",
    "ay_uc_t_lead3",
    "gx_t_lead3",
    "gy_t_lead3",
    "ar_lag1",
    "gr_lag1",
    "ar_uc_lag1",
    "az_e_lag1",
    "az_uc_e_lag1",
    "gz_e_lag1",
    "ax_t_lag1",
    "ay_t_lag1",
    "ax_uc_t_lag1",
    "ay_uc_t_lag1",
    "gx_t_lag1",
    "gy_t_lag1",
    "ar_lag2",
    "gr_lag2",
    "ar_uc_lag2",
    "az_e_lag2",
    "az_uc_e_lag2",
    "gz_e_lag2",
    "ax_t_lag2",
    "ay_t_lag2",
    "ax_uc_t_lag2",
    "ay_uc_t_lag2",
    "gx_t_lag2",
    "gy_t_lag2",
    "ar_lag3",
    "gr_lag3",
    "ar_uc_lag3",
    "az_e_lag3",
    "az_uc_e_lag3",
    "gz_e_lag3",
    "ax_t_lag3",
    "ay_t_lag3",
    "ax_uc_t_lag3",
    "ay_uc_t_lag3",
    "gx_t_lag3",
    "gy_t_lag3",
]


df = pd.read_parquet("../data/processed/2025-03-29_23-55-46_5Hz.parquet")

df.info()

df.head(20)

df_info_to_file(
    df,
    file_name="df_info.txt",
    include_dtypes=True,
    include_desc_stats=True,
    include_nulls=True,
    include_sample=True,
    sample_rows=5,
)


labels, kmeans_model = apply_kmeans(df, features, n_clusters=5)
visualize_kmeans(df, features, labels, kmeans_model)


tsne_result, tsne_model = apply_tsne(df, features, perplexity=30)
visualize_tsne(df, features, tsne_result, color_col="speed")


df_plot = df.copy()[["elapsed_seconds"] + features].dropna()
df_plot["kmeans5"] = labels


plot_variable_by_cluster(
    df_plot,
    var_col="gy_t",
    cluster_col="kmeans5",
    time_col="elapsed_seconds",
    marker_size=7,
    opacity=0.8,
    title=None,
)

# ------------------------------------------

df_with_jumps = identify_jumps(
    df, accel_col="az_uc_e", threshold=1.5, new_col="jump", inplace=False
)

df_with_jumps

plot_variable_by_cluster(
    df_with_jumps,
    var_col="az_uc_e",
    cluster_col="jump",
    time_col="elapsed_seconds",
    marker_size=7,
    opacity=0.8,
    title="Vertical Acceleration with Jump Detection",
)

# ------------------------------------------

# Developing jump events function:


def extract_jump_events(
    df: pd.DataFrame,
    jump_col: str = "jump",
    time_col: str = "elapsed_seconds",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    speed_col: str = "speed",
    min_consecutive: int = 2,
) -> pd.DataFrame:
    """
    NOTE: Consider modifying so time deltas calculated from DatetimeIndex
    and not from elapsed_seconds.

    Extract jump events from the dataframe by identifying consecutive jump points.

    Parameters:
        df (pd.DataFrame): Input DataFrame with jump indicators
        jump_col (str): Column name containing jump indicators (1 for jump, 0 for no jump)
        time_col (str): Column name for time data (seconds)
        lat_col (str): Column name for latitude
        lon_col (str): Column name for longitude
        distance_col (str): Column name for cumulative distance
        min_consecutive (int): Minimum number of consecutive jump points to qualify as a jump event

    Returns:
        pd.DataFrame: DataFrame with extracted jump events and their properties
    """
    # Check required columns exist
    required_cols = [jump_col, time_col]
    optional_cols = [lat_col, lon_col, speed_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in DataFrame")

    # Check which optional columns exist
    available_cols = {}
    for col in optional_cols:
        available_cols[col] = col in df.columns

    # Create a copy with only necessary columns
    df_work = df.copy()

    # Create a group identifier for consecutive jump points
    # This trick identifies runs of consecutive 1s by looking at when the values change
    df_work["jump_change"] = df_work[jump_col].diff().ne(0).cumsum()

    # Group by this identifier to find consecutive sequences
    jump_groups = df_work[df_work[jump_col] == 1].groupby("jump_change")

    # Initialize a list to store jump event data
    jump_events = []

    # Process each jump group
    for _, group in jump_groups:
        # Skip if fewer than min_consecutive points
        if len(group) < min_consecutive:
            continue

        # Calculate basic properties
        start_time = group[time_col].iloc[0]
        end_time = group[time_col].iloc[-1]
        airtime_s = end_time - start_time

        # Initialize event data
        event = {
            "elapsed_seconds_ride": start_time,
            "airtime_s": airtime_s,
            "points": len(group),
        }

        if available_cols[speed_col]:
            speed_avg = group[speed_col].mean()
            event["speed_mph"] = speed_avg * 2.23694  # Convert m/s to mph
            event["distance_ft"] = (
                airtime_s * event["speed_mph"] * 1.4667
            )  # Convert mph to ft/s

        # Add location data if available (just the starting point)
        if available_cols[lat_col] and available_cols[lon_col]:
            event["latitude"] = group[lat_col].iloc[0]
            event["longitude"] = group[lon_col].iloc[0]

        # Use DataFrame's DatetimeIndex for timestamps if available
        if isinstance(df.index, pd.DatetimeIndex):
            event["datetime"] = group.index[0]

        # Add event to the list
        jump_events.append(event)

    if not jump_events:
        print("No jump events found.")
        return None

    # Create the jumps DataFrame
    jumps_df = pd.DataFrame(jump_events)

    # Sort by start time
    jumps_df = jumps_df.sort_values("elapsed_seconds_ride").reset_index(drop=True)

    return jumps_df


# diff in seconds from datetimeindex:
(df.index[1] - df.index[0]).total_seconds()

jump_events_df = extract_jump_events(
    df_with_jumps, jump_col="jump", time_col="elapsed_seconds", min_consecutive=2
)

jump_events_df


# ------------------------------------------

analyze_pca_components(df, features, standardize=True, plot=True, verbose=False)

if __name__ == "__main__":
    pass


# Load your data
# df = pd.read_parquet('../data/processed/2025-03-29_23-55-46_5Hz.parquet')

# Define features for clustering
# These will be the motion sensor columns that are most relevant
# features = ['ax_t', 'ay_t', 'az_t', 'gx_t', 'gy_t', 'gz_t', 'speed']

# Apply and visualize KMeans
# labels, kmeans_model = apply_kmeans(df, features, n_clusters=5)
# visualize_kmeans(df, features, labels, kmeans_model)

# Apply and visualize DBSCAN
# labels, dbscan_model = apply_dbscan(df, features, eps=0.5, min_samples=5)
# visualize_dbscan(df, features, labels, dbscan_model)

# Apply and visualize t-SNE
# tsne_result, tsne_model = apply_tsne(df, features, perplexity=30)
# visualize_tsne(df, features, tsne_result, color_col='speed')

# Apply and visualize HMM
# states, hmm_model = apply_hmm(df, features, n_components=4)
# visualize_hmm(df, features, states, hmm_model)
