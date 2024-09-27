# Customer Segmentation using K-Means Clustering

## Problem Statement
The goal of this project is to help a business understand its customers better by segmenting them into distinct groups. 
Customer segmentation enables businesses to tailor marketing strategies and make data-driven decisions to improve customer satisfaction and maximize profitability. 
Using K-Means clustering, we aim to classify customers into different segments based on their purchasing behavior and demographics. 
This will allow the business to target specific groups with personalized marketing, thereby increasing efficiency and customer retention.

## Dataset
The dataset used in this project is sourced from Kaggle: Customer Segmentation Dataset. 
The data consists of basic customer attributes:

- Customer ID: Unique identifier for each customer.
- Gender: Male or Female.
- Age: Age of the customer.
- Annual Income: Income in thousands of dollars.
- Spending Score: A score assigned by the mall based on customer behavior and purchasing patterns (scale of 1-100).

# Workflow

## Data Preprocessing:

- File Upload: The user uploads a CSV file containing customer data via the Streamlit web app.
- Data Cleaning: The app checks for missing values and alerts the user if there are any issues that need to be handled.
- Feature Selection: The user selects numeric columns (e.g., Annual Income, Spending Score) for clustering within the app interface.
- Scaling: The selected features are scaled using StandardScaler to ensure that all variables are on the same scale.

## Model Development:

#### A K-Means clustering algorithm is used to segment customers into different clusters.The model is built using the scikit-learn library, and two options are available:

- Pre-Trained Model: A previously trained K-Means model (kmeans_model.pkl) is loaded and applied to the uploaded data.
- New Model: The user can opt to train a new K-Means model with a custom number of clusters.
- WCSS (Within-Cluster Sum of Squares) is used to find the optimum number of clusters using the Elbow Method.
- Silhouette Score is used to evaluate the quality of the clusters.

## Visualization:

- Cluster Visualization: Clusters are visualized using Plotly, showing a scatter plot of the data with centroids.
- Distribution Plots: Seaborn is used to create histograms that show the distribution of the selected numeric features across the clusters.
- The app displays a cluster summary table that shows the mean values for each feature in every cluster.

## Output and Download:

- The clustered dataset is displayed with an additional column indicating the cluster labels for each customer.
- Users can download the clustered dataset in CSV format directly from the web app.
- A cluster summary showing the mean values of the numeric features in each cluster is also available for download.

# Technologies Used
### Python: The primary programming language used for the entire project.
#### Libraries:
- Streamlit: For building the interactive web application that allows users to upload data, view visualizations, train models, and download results.
- pandas: For data manipulation and preprocessing.
- numpy: For numerical operations.
- matplotlib & seaborn: For data visualization and plotting histograms.
- plotly: For interactive visualizations of the clusters.
- scikit-learn: For implementing the K-Means clustering algorithm and scaling features.
- joblib: For loading and saving the pre-trained model.

# Deployment:
The project was successfully deployed on both a local server and Streamlit Cloud for broader accessibility, enabling businesses to easily access the app for real-time customer segmentation.

## Streamlit Web App:
#### Users can either use a pre-trained model or train a new model directly within the app.

- Pre-Trained Model: The app loads and applies a pre-trained K-Means model to the uploaded data.
- New Model: Users can train a new model by selecting the number of clusters interactively within the app.
- The Elbow Method and Silhouette Score are used to help users determine the best number of clusters.
  
## Key Features
- User-Friendly Interface: The application allows users to upload their own data and interactively select features for clustering.
 This makes it easy for non-technical users to apply the K-Means clustering algorithm to their data.
- Model Flexibility: Users can either apply a pre-trained model for instant results or train a new model based on their own input.
- Visual Insights: The app provides visual insights into the clusters through scatter plots, histograms, and cluster summaries, helping businesses to easily understand their customer segments.
- Downloadable Results: After clustering, the user can download the clustered data and cluster summary, which can be used for further analysis or reporting.

# Challenges and Solutions
- Scaling: The numeric features must be scaled before applying the K-Means algorithm. This was handled by using StandardScaler to normalize the input features,
  ensuring that the clustering results are not skewed by varying scales.
- Cluster Evaluation: Determining the optimal number of clusters is often tricky. We used the Elbow Method and Silhouette Score to guide the selection of an appropriate number of clusters.
- User Interaction: Building an intuitive interface for data uploading and parameter selection was key. Streamlit's widgets and Plotly visualizations made it easier to create a smooth user experience.


# Conclusion
This customer segmentation project effectively demonstrates how machine learning, specifically K-Means clustering, can help businesses better understand their customer base and tailor their marketing strategies. 
The project showcases the integration of a machine learning model into a user-friendly web application using Streamlit, making it accessible to a wide range of users who may not have a technical background. 
Analyzing key customer attributes, the business can make informed decisions that lead to more personalized marketing efforts and improved customer satisfaction.
