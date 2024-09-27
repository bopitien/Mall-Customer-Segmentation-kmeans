import numpy as np
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the pre-trained model (the model you've trained)
pretrained_model_path = "kmeans_model.pkl"
kmeans_pretrained = joblib.load(pretrained_model_path)

# Define the data processing pipeline (scaling)
def process_data(df, features):
    """
    This function selects the numeric features specified by the user and scales them using StandardScaler.
    It returns the scaled version of the selected features.
    """
    X = df[features]  # Select the relevant numeric columns
    scaler = StandardScaler()  # Create a StandardScaler instance
    X_scaled = scaler.fit_transform(X)  # Scale the data
    return X_scaled

# App title
st.title("Customer Segmentation with K-Means")

# Upload the CSV data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())  # Display a preview of the uploaded data
    
    # Check for missing values and alert the user if any exist
    if df.isnull().values.any():
        st.error("Your dataset contains missing values. Please clean the data or handle missing values before proceeding.")
    else:
        # Allow users to select which numeric columns they want to include in the clustering
        st.write("Select the numeric columns for clustering:")
        st.write("Columns in uploaded data:", df.columns)
        numeric_cols = st.multiselect(
            "Select Columns", 
            options=df.select_dtypes(include=[np.number]).columns.tolist(), 
            default=["Annual Income (k$)", "Spending Score (1-100)"]
        )

        if numeric_cols:
            # Process the data (scale the selected columns)
            X_scaled = process_data(df, numeric_cols)

            # Ask the user if they want to use the pre-trained model or train a new one
            st.write("Do you want to use the pre-trained model or train a new model?")
            model_choice = st.radio("Choose an option", ('Use Pre-Trained Model', 'Train a New Model'))

            if model_choice == 'Use Pre-Trained Model':
                # Use the pre-trained model (your trained model loaded earlier)
                st.write("Using the pre-trained model for clustering...")
                
                # Predict clusters using the pre-trained model
                cluster_labels = kmeans_pretrained.predict(X_scaled)
                df['Cluster'] = cluster_labels  # Add the cluster labels to the original DataFrame
                
                # Display cluster mean values and visualization
                st.write("Mean values for each cluster:")

                
                # Get numeric columns, but exclude 'CustomerID'
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'CustomerID' in numeric_columns:
                    numeric_columns.remove('CustomerID')  # Remove 'CustomerID' from the list of numeric columns
                # Group by 'Cluster' and calculate means for only numeric columns
                cluster_means = df.groupby('Cluster')[numeric_columns].mean()  # Group by 'Cluster' and calculate means for numeric columns
                st.write(cluster_means)

                # Plot clusters using Plotly for interactivity
                st.write("Cluster Visualization with Centroids:")
                centroids = kmeans_pretrained.cluster_centers_  # Get the centroids of the clusters
                fig = px.scatter(
                    x=X_scaled[:, 0], 
                    y=X_scaled[:, 1], 
                    color=cluster_labels.astype(str), 
                    title="Cluster Visualization",
                    labels={ "x": numeric_cols[0], "y": numeric_cols[1] }
                )
                # Add centroids to the plot
                fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(size=12, color='red'), name='Centroids')
                st.plotly_chart(fig)


                # Distribution of selected numeric columns
                st.write("Distribution of Numeric Columns:")
                for col in numeric_cols:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[col], kde=True, bins=20)
                    plt.title(f"Distribution of {col}")
                    st.pyplot(plt)

                # Allow users to download the dataset with cluster labels
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV with Cluster Labels",
                    data=csv,
                    file_name='clustered_data.csv',
                    mime='text/csv',
                )

                # Allow users to download the cluster means.... (summary)
                cluster_summary_csv = cluster_means.to_csv(index=False)
                st.download_button(
                    label="Download Cluster Summary",
                    data=cluster_summary_csv,
                    file_name='cluster_summary.csv',
                    mime='text/csv',
                )

            else:
                # Train a new K-Means model.....
                st.write("Training a new model...")

                # Plot Elbow Method to guide users in selecting the number of clusters
                def plot_elbow(X_scaled):
                    wcss = []
                    for i in range(1, 11):  # Iterate over cluster sizes from 1 to 10
                        kmeans_test = KMeans(n_clusters=i, init='k-means++', random_state=42)
                        kmeans_test.fit(X_scaled)
                        wcss.append(kmeans_test.inertia_)  # Store the WCSS value for each number of clusters
                    plt.figure(figsize=(8, 5))
                    plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
                    plt.title('Elbow Method')
                    plt.xlabel('Number of Clusters')
                    plt.ylabel('WCSS')
                    st.pyplot(plt)

                st.write("Elbow Method to guide cluster selection:")
                plot_elbow(X_scaled)  # Display the Elbow plot

                # Allow the user to select the number of clusters (based on Elbow Method)
                n_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=4)

                # Fit the K-Means model using the selected number of clusters
                kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
                kmeans.fit(X_scaled)
                cluster_labels = kmeans.predict(X_scaled)
                df['Cluster'] = cluster_labels  # Add the cluster labels to the original DataFrame

                # Display mean values for each cluster
                st.write("Mean values for each cluster:")

                # Get numeric columns, but exclude 'CustomerID' (and any other non-numeric columns like 'Gender')
                numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
                if 'CustomerID' in numeric_columns:
                    numeric_columns.remove('CustomerID')  # Remove 'CustomerID' if it's in the numeric columns list

                # Group by 'Cluster' and calculate means for only numeric columns
                cluster_means = df.groupby('Cluster')[numeric_columns].mean()
                st.write(cluster_means)

                # Display evaluation metrics (Silhouette Score and WCSS)
                silhouette_avg = silhouette_score(X_scaled, cluster_labels)
                wcss = kmeans.inertia_
                st.write(f"Silhouette Score: {silhouette_avg:.4f}")
                st.write(f"Inertia (WCSS): {wcss:.4f}")

                # Plot clusters using Plotly for interactivity
                st.write("Cluster Visualization with Centroids:")
                centroids = kmeans.cluster_centers_  # Get the centroids of the clusters
                fig = px.scatter(
                    x=X_scaled[:, 0], 
                    y=X_scaled[:, 1], 
                    color=cluster_labels.astype(str), 
                    title="Cluster Visualization",
                    labels={ "x": numeric_cols[0], "y": numeric_cols[1] }
                )
                # Add centroids to the plot
                fig.add_scatter(x=centroids[:, 0], y=centroids[:, 1], mode='markers', marker=dict(size=12, color='red'), name='Centroids')
                st.plotly_chart(fig)

                # Distribution of selected numeric columns
                st.write("Distribution of Numeric Columns:")
                for col in numeric_cols:
                    plt.figure(figsize=(8, 6))
                    sns.histplot(df[col], kde=True, bins=20)
                    plt.title(f"Distribution of {col}")
                    st.pyplot(plt)

                # Allow users to download the dataset with cluster labels
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV with Cluster Labels",
                    data=csv,
                    file_name='clustered_data.csv',
                    mime='text/csv',
                )

                # Allow users to download the cluster means (summary)
                cluster_summary_csv = cluster_means.to_csv(index=False)
                st.download_button(
                    label="Download Cluster Summary",
                    data=cluster_summary_csv,
                    file_name='cluster_summary.csv',
                    mime='text/csv',
                )
