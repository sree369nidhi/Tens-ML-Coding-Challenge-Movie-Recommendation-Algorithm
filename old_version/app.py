# This is the main application file. It contains the Streamlit app and the login functionality.


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
from pyspark.ml.recommendation import ALSModel

# Dummy user database
users = {
    "admin": "password123",
    "user1": "pass1",
}



def main_app():

    @st.cache_data
    def load_and_preprocess_data_balanced():
        # Load data
        movies = pd.read_csv('data/movies.csv')
        ratings = pd.read_csv('data/ratings.csv')
        # Merge datasets on movieId
        merged_data = pd.merge(ratings, movies, on='movieId')
        train_data = pd.DataFrame()
        test_data = pd.DataFrame()
        for user_id in merged_data['userId'].unique():
            user_data = merged_data[merged_data['userId'] == user_id]
            # Adjust split ratio if number of ratings is odd
            # Favor training set by allocating the extra review to it
            split_ratio = 0.5 if len(user_data) % 2 == 0 else (len(user_data) // 2 + 1) / len(user_data)
            user_train, user_test = train_test_split(user_data, test_size=1-split_ratio, shuffle=True)
            train_data = pd.concat([train_data, user_train])
            test_data = pd.concat([test_data, user_test])
        return merged_data, train_data, test_data

    @st.cache_data
    def calculate_sparsity(dataframe, user_col='userId', item_col='movieId'):
        # Count the total number of possible interactions (num_users * num_items)
        num_users = dataframe[user_col].nunique()
        num_items = dataframe[item_col].nunique()
        total_possible_interactions = num_users * num_items

        # Count the number of actual interactions
        num_actual_interactions = dataframe.shape[0]

        # Calculate sparsity
        sparsity = 1 - (num_actual_interactions / total_possible_interactions)
        return sparsity


    # Load and preprocess data
    data1, train_data_balanced, test_data_balanced = load_and_preprocess_data_balanced()
    train_sparsity = calculate_sparsity(train_data_balanced)
    test_sparsity = calculate_sparsity(test_data_balanced)

    # Initialize Spark Session
    spark = SparkSession.builder.appName("MovieRecommender").getOrCreate()

    # Convert DataFrames to Spark DataFrames
    train_data_spark = spark.createDataFrame(train_data_balanced)
    test_data_spark = spark.createDataFrame(test_data_balanced)


    # This function generates predictions for the entire test dataset at once and caches the result.
    @st.cache_data
    def train_and_generate_predictions(test_data):

        # # Train ALS model
        als = ALS(userCol="userId", itemCol="movieId", ratingCol="rating", coldStartStrategy="drop", nonnegative=True, rank=30, maxIter=10, regParam=0.1)
        model = als.fit(train_data_spark)

        # Convert the test DataFrame to a Spark DataFrame here, ensuring all operations leading to the prediction are cached.
        test_data_spark = spark.createDataFrame(test_data)
        
        # Generate predictions
        predictions = model.transform(test_data_spark)
        return predictions.toPandas()

    # Convert predictions to Pandas DataFrame for easier handling in Streamlit
    pred_pandas = train_and_generate_predictions(test_data_balanced)

    pred_pandas['prediction'] = pred_pandas['prediction'].apply(lambda x: min(x, 5.0))
    pred_pandas.rename(columns={'prediction': 'ALS_predicted'}, inplace=True)

    # Streamlit App
    st.title("Movie Recommendation System")

    # Sidebar Navigation
    page = st.sidebar.selectbox("Navigate", ["Data Overview", "Feature Enhancement", "Test Train Split Overview", "Recommendation Abstract", "Recommendation Demo"])

    if page == "Data Overview":
        st.header("Data Overview")
        st.write("Movies DataFrame")
        st.dataframe(data1[['movieId', 'title']].drop_duplicates().head())
        st.write("Ratings DataFrame")
        st.dataframe(data1[['userId', 'movieId', 'rating']].head())

    elif page == "Feature Enhancement":
        st.header("Feature Enhancement")
        st.write("For any recommendation engine, data should be in the form of a matrix, not a DataFrame. Here, users are rows and movies are columns.")
        if os.path.exists('/path/to/matrix_image.png'):
            st.image('/path/to/matrix_image.png', caption='User-Item Matrix')
        else:
            st.write("Matrix image not found.")
        st.write(f"Train dataset sparsity: {train_sparsity:.4f}")
        st.write(f"Test dataset sparsity: {test_sparsity:.4f}")

    elif page == "Test Train Split Overview":
        st.header("Test Train Split Overview")
        st.code("""
    for user_id in merged_data['userId'].unique():
        user_data = merged_data[merged_data['userId'] == user_id]
        split_ratio = 0.5 if len(user_data) % 2 == 0 else (len(user_data) // 2 + 1) / len(user_data)
        user_train, user_test = train_test_split(user_data, test_size=1-split_ratio, shuffle=True)
    """, language='python')

    elif page == "Recommendation Abstract":
        st.header("Recommendation Abstract")
        st.write("""
    This engine identifies patterns among users' data to recommend movies. It's based on Collaborative Filtering, specifically using ALS (Alternating Least Squares), a matrix factorization technique that deals with the user-item interactions.

    Advantages of ALS include scalability, handling of sparse datasets, and the ability to incorporate implicit feedback. We chose ALS over KNN due to its efficiency and effectiveness in handling large datasets.

    ALS RMSE: 0.9191026086542822
    KNN RMSE: 1.7873672429944476
    Based on RMSE, ALS provides more accurate predictions, making it the selected model for our recommendation system.
    """)

    elif page == "Recommendation Demo":
        st.header("Recommendation Demo")
        user_id = st.number_input("Enter a user ID", min_value=1, value=1, step=1)

        if st.button("Show Recommendations"):
            st.subheader(f"Movies rated by User ID {user_id} in the training data:")
            user_train_ratings = train_data_balanced[train_data_balanced['userId'] == user_id]
            st.dataframe(user_train_ratings[['movieId', 'title', 'rating']].sort_values(by='rating', ascending=False).head())

            st.subheader(f"Top 5 movie recommendations and predicted rating for User ID {user_id}:")
            user_recommendations = pred_pandas[pred_pandas['userId'] == user_id].nlargest(5, 'ALS_predicted')
            st.dataframe(user_recommendations[['movieId', 'title', 'ALS_predicted']])

            st.subheader(f"Evaluation Metrics for User ID {user_id}:")
            user_actual_vs_predicted = pred_pandas[pred_pandas['userId'] == user_id]
            user_actual_vs_predicted['error'] = np.abs(user_actual_vs_predicted['rating'] - user_actual_vs_predicted['ALS_predicted'])

            # Error statistics
            sum_error = user_actual_vs_predicted['error'].sum()
            mean_absolute_error = user_actual_vs_predicted['error'].mean()
            rmse = np.sqrt((user_actual_vs_predicted['error']**2).mean())

            significant_error_threshold = 1.5  #  threshold for significant prediction error
            significant_errors = user_actual_vs_predicted[user_actual_vs_predicted['error'] > significant_error_threshold]
            false_positives = significant_errors[significant_errors['ALS_predicted'] > significant_errors['rating']].shape[0]
            false_negatives = significant_errors[significant_errors['ALS_predicted'] < significant_errors['rating']].shape[0]
            # Calculate True Positives (TP): Predictions within the significant error threshold
            true_positives = user_actual_vs_predicted[user_actual_vs_predicted['error'] <= significant_error_threshold].shape[0]

            # Note: In this context, we're only considering predictions close to actual ratings as "accurate."
            total_predictions = user_actual_vs_predicted.shape[0]
            accuracy = true_positives / total_predictions if total_predictions else 0

            st.write(f"Sum of Errors/Total error: {sum_error:.2f}")
            st.write(f"Mean Absolute Error: {mean_absolute_error:.4f}")
            st.write(f"Root Mean Squared Error: {rmse:.4f}")
            st.write(f"Accuracy: {accuracy:.2%}")
            st.write(f"False Positives: {false_positives}")
            st.write(f"False Negatives: {false_negatives}")
            st.write(f"True Positives: {true_positives}")


            # Display all predictions for the user with scroll bar
            st.subheader(f"All movie predictions for User ID {user_id}:")
            st.dataframe(user_actual_vs_predicted[['movieId', 'title', 'rating', 'ALS_predicted', 'error']].sort_values(by='error', ascending=True))

            





# Assume your other code parts here...

def check_login(username: str, password: str):
    # Function to validate credentials
    return username in users and users[username] == password

def login_user():
    # Simplified login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username and password:  # Check if both fields are not empty
            if check_login(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.experimental_rerun()  # Trigger rerun after successful login
            else:
                st.error("Invalid username or password")
        else:
            st.warning("Please enter both username and password")

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    main_app()
else:
    st.title("Login to the Movie Recommendation System")
    login_user()
