import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set page configuration
st.set_page_config(page_title="Customer Segmentation", layout="wide")
st.title("Segmentation Dashboard")

st.sidebar.title("Navigation")
st.sidebar.markdown("## Select an option")
options = ["Data Upload", "Data Exploration", "Clustering", "Visualization"]
selected_option = st.sidebar.selectbox("Choose an option", options)

# Initialize session state variables

#Change page title based on selected option
if selected_option == "Data Upload":
    st.title("Upload Your Dataset")
    st.markdown("Upload a CSV file containing customer data for segmentation.")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully!")
        st.dataframe(df.head())
        st.session_state.df = df  # Store the DataFrame in session state
        st.session_state.uploaded = True

if selected_option == "Data Exploration":
    if 'uploaded' not in st.session_state or not st.session_state.uploaded:
        st.warning("Please upload a CSV file first.")
    else:
        st.title("Data Exploration")
        df = st.session_state.df  # Retrieve the DataFrame from session state
        st.markdown("### Data Overview")
        st.dataframe(df.head())

        # Display basic statistics
        st.markdown("### Descriptive Statistics")
        st.write(df.describe())

        # Display data types
        st.markdown("### Data Types")
        st.write(df.dtypes)

        st.markdown("### Data Shape")
        st.write(f"Number of rows: {df.shape[0]}")


        # Display missing values
        st.markdown("### Missing Values")
        st.write(df.isnull().sum())

if selected_option == "Clustering":
    if 'uploaded' not in st.session_state or not st.session_state.uploaded:
        st.warning("Please upload a CSV file first.")
    else:
        st.title("Clustering")
        df = st.session_state.df  # Retrieve the DataFrame from session state

        # Select features for clustering
        st.markdown("### Select Features for Clustering")
        features = st.multiselect("Select features", df.columns.tolist()) # remember the features are the columns
        if len(features) < 2:
            st.warning("Please select at least two features for clustering.")
        else:
            st.write(df[features].head())  # Display the selected features
            # Check if the selected features are numeric
            numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_features) < 2:
                st.warning("Please select at least two numeric features for clustering.")
            else:
                # Check if the selected features are numeric
                numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
                if len(numeric_features) < 2:
                    st.warning("Please select at least two numeric features for clustering.")
                else:
                    # Display the selected features
                    st.write(df[numeric_features].head())
            # Standardize the data (important for KMeans: this ensures that all features contribute equally to the distance calculations)
            #also remember that with KMeans, the data should be numeric
            # Check if the selected features are numeric
            # numeric_features = df[features].select_dtypes(include=[np.number]).columns.tolist()
            st.markdown("### Data Standardization")
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df[features])

            # Let's use elbow method to determine the optimal number of clusters
            st.markdown("### Elbow Method")
            inertias =[]
            k_values = range(1, 11)

            for k in k_values:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
        
            fig, ax = plt.subplots()
            ax.set_title("Elbow Method for Optimal k")
            ax.set_xlabel("Number of Clusters (k)")
            ax.set_ylabel("Inertia")

           
            sns.lineplot(x=list(k_values), y=inertias, markers='o', ax=ax)
            plt.xticks(k_values)
            st.pyplot(fig)
    
            st.markdown("### Optimal Number of Clusters")


            # KMeans clustering
            st.markdown("### KMeans Clustering")
            # Automatically determine the optimal number of clusters using the elbow method
            second_derivative = np.diff(np.diff(inertias))  # Compute second-order differences
            optimal_k = np.argmax(second_derivative) + 4   # +2 because np.diff() reduces size twice

            st.write(f"Optimal number of clusters determined by the elbow method: {optimal_k}")
            k = st.slider("Select number of clusters", 2, 10, value=optimal_k)  # Default to optimal_k
            kmeans = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = kmeans.fit_predict(scaled_data)


         

            # Display clustered data
            st.markdown("### Clustered Data")
            st.dataframe(df.head())

            st.session_state.clustered_df = df  # Store the clustered DataFrame in session state
            st.session_state.k = k  # Store the number of clusters in session state

        

            #kmeans Cluster Centers
            st.markdown("### KMeans Cluster Centers")
            clus_df = pd.DataFrame(kmeans.cluster_centers_, columns=features)
            st.dataframe(clus_df)



            #feature importance using centroids_min and centroids_max
            st.markdown("### Feature Importance")
            centroids_ranges = pd.DataFrame({
                'Feature': features,
                'Min': kmeans.cluster_centers_.min(axis=0),
                'Max': kmeans.cluster_centers_.max(axis=0)
            })
            centroids_ranges['Importance'] = centroids_ranges['Max'] - centroids_ranges['Min']
            centroids_ranges = centroids_ranges.sort_values(by='Importance', ascending=False) 
            st.dataframe(centroids_ranges)
            st.dataframe(centroids_ranges.select_dtypes(include=['number']).std())


            #value counts of clusters per cluster and feature
            st.markdown("### Value Counts of Clusters")
            cluster_counts = df['Cluster'].value_counts().reset_index()
            cluster_counts.columns = ['Features','Cluster', 'Count']
            st.dataframe(cluster_counts)
        
               #Group data by cluster and calculate mean values for each cluster
            st.markdown("### Cluster Counts within Each Feature")

            summary_table = df.groupby('Cluster')[features].count().T
            st.dataframe(summary_table)



if selected_option == "Visualization":
    if 'clustered_df' not in st.session_state:
        st.warning("Please perform clustering first.")
    else:
        st.title("Visualization")
        df = st.session_state.clustered_df  # Retrieve the clustered DataFrame from session state
        k = st.session_state.k  # Retrieve the number of clusters from session state

        # Select features for visualization
        st.markdown("### Select Features for Visualization")
        features = st.multiselect("Select features", df.columns.tolist(), default=df.columns.tolist()[:2])
        if len(features) < 2:
            st.warning("Please select at least two features for visualization.")
        else:
            # Scatter plot of the selected features colored by cluster
            fig, ax = plt.subplots()
            ax.set_title("Scatter Plot of Selected Features")
            sns.scatterplot(data=df, x=features[0], y=features[1], hue='Cluster', palette='Set1', ax=ax)
            st.pyplot(fig)
