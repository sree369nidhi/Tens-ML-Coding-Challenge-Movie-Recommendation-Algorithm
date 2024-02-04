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

from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import bcrypt


Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

engine = create_engine('sqlite:///users.db', echo=True)
Base.metadata.create_all(engine)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db_session():
    return SessionLocal()

def login_user():
    st.title("Login to the Movie Recommendation System")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        db = get_db_session()  # Adjusted session handling
        if username and password:
            if check_login(db, username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                db.close()  # Close the session after use
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
                db.close()  # Close the session after use
        else:
            st.warning("Please enter both username and password")

def registration_form():
    st.title("Register for the Movie Recommendation System")
    username = st.text_input("Choose a Username", key="reg_username")
    password = st.text_input("Choose a Password", type="password", key="reg_password")
    
    if st.button("Register"):
        db = get_db_session()  # Adjusted session handling
        if username and password:
            if db.query(User).filter(User.username == username).first():
                st.error("Username already exists. Please choose a different one.")
                db.close()  # Close the session after use
            else:
                register_user(db, username, password)
                st.success("Registration successful. Please login.")
                db.close()  # Close the session after use
        else:
            st.warning("Please enter both username and password")


# def hash_password(password):
#     return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def hash_password(password):
    print("Password type:", type(password))  # Should be <class 'str'>
    password_bytes = password.encode('utf-8')
    print("Password bytes type:", type(password_bytes))  # Should be <class 'bytes'>
    salt = bcrypt.gensalt()
    hashed_password = bcrypt.hashpw(password_bytes, salt)
    return hashed_password.decode('utf-8')


def check_password(hashed_password, password):
    return bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8'))

def check_login(db, username, password):
    user = db.query(User).filter(User.username == username).first()
    if user and check_password(user.password_hash, password):
        return True
    return False

def register_user(db, username, password):
    hashed_password = hash_password(password)
    new_user = User(username=username, password_hash=hashed_password)
    db.add(new_user)
    db.commit()


# Main application

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

    # Sidebar navigation with an added 'About' page
    page = st.sidebar.selectbox(
        "Navigate",
        [
            "Data Overview",
            "Feature Enhancement",
            "Test Train Split Overview",
            "Recommendation Abstract",
            "Recommendation Demo",
            "Extra All Predictions",

        ])


    if page == "Data Overview":
        st.header("Data Overview")
        st.write("Here's a brief overview of the initial raw data used in our recommendation system.")
        
        # Display Movies DataFrame
        st.subheader("Movies DataFrame")
        st.write("This DataFrame contains movie IDs and titles, representing the movies available for recommendation.")
        st.dataframe(data1[['movieId', 'title']].drop_duplicates().head())
        
        # Display Ratings DataFrame
        st.subheader("Ratings DataFrame")
        st.write("This DataFrame includes user IDs, movie IDs, and ratings, reflecting user interactions with movies.")
        st.dataframe(data1[['userId', 'movieId', 'rating']].head())



    elif page == "Feature Enhancement":
        st.header("Feature Enhancement")
        st.write("""
                 
        For any recommendation engine, data should be in the form of a matrix, not a DataFrame. Here, users are rows and movies are columns. This structure helps in identifying patterns and making recommendations based on similarities.

        To build this matrix on our data, we consider users as rows and movies as columns. Then, we check the sparsity of the matrix to assess data suitability for modeling. No additional columns are created, as we need only 3 columns (user ID, movie ID, rating) for building this recommendation engine.

        In recommendation systems, it is common to encounter sparse datasets where the majority of possible user-item interactions are unknown or unrecorded. This sparsity is a measure of the proportion of potential user-item interactions that have not been observed:

        - A high sparsity indicates that we have a vast number of potential recommendations that the model can make. It challenges the model to infer preferences from a limited number of interactions.
        - A low sparsity suggests that users have interacted with a significant portion of the item catalog, which could make it easier for the model to identify preferences but may also reduce the necessity for complex recommendation algorithms.

        In our case, the data sparsity levels indicate that we are working with a typical user-item interaction dataset where most user-movie combinations have not been rated. This justifies the use of sophisticated modeling techniques like collaborative filtering to predict these missing ratings effectively.

        The matrix form of the dataset, required for our recommendation engine, is built by considering users as rows and movies as columns. The sparsity of this matrix is calculated as follows:
        """)
        
        # Display matrix image if exists
        if os.path.exists('/path/to/matrix_image.png'):
            st.image('/path/to/matrix_image.png', caption='User-Item Matrix')
        else:
            st.write("Matrix image not found.")
        
        # Display sparsity
        st.write(f"Train dataset sparsity: {train_sparsity:.4f}")
        st.write(f"Test dataset sparsity: {test_sparsity:.4f}")
        st.write("""
        These sparsity figures help us understand the scope of our recommendation system's challenge. A sparsity level close to 1 means that we have very few ratings compared to the number of possible user-movie pairs, which is common in real-world scenarios. Our recommendation system aims to fill these gaps by predicting the likely ratings users would give to movies they have not yet seen.
        """)



    elif page == "Test Train Split Overview":
        st.header("Test Train Split Overview")
        st.write("""
        The division of data into training and testing sets is a crucial step in evaluating the performance of our recommendation engine. To ensure that all users are represented in both datasets, we perform a balanced split:

        - For users with an even number of ratings, we equally distribute their ratings between the train and test sets.
        - For users with an odd number of ratings, we assign the majority to the training set. This approach slightly favors the training data, providing it with a more comprehensive set of information for model training, which is essential for a recommendation system that learns from user behavior.

        Below is the high-level code used for splitting the dataset:
        """)
        
        # Display code for test train split
        st.code("""
        for user_id in merged_data['userId'].unique():
            user_data = merged_data[merged_data['userId'] == user_id]
            split_ratio = 0.5 if len(user_data) % 2 == 0 else (len(user_data) // 2 + 1) / len(user_data)
            user_train, user_test = train_test_split(user_data, test_size=1-split_ratio, shuffle=True)
                
        # Verify the split for a few users
        for user_id in train_data_balanced['userId'].unique()[:5]:  # Check for the first 5 users
            train_reviews = train_data_balanced[train_data_balanced['userId'] == user_id]
            test_reviews = test_data_balanced[test_data_balanced['userId'] == user_id]
            diff = abs(len(train_reviews) - len(test_reviews))
            assert diff <= 1, f"User {user_id} does not have an equal or nearly equal split"
            print(f"User {user_id}: Train reviews = {len(train_reviews)}, Test reviews = {len(test_reviews)}")
                
        User 1: Train reviews = 116, Test reviews = 116
        User 5: Train reviews = 22, Test reviews = 22
        User 7: Train reviews = 76, Test reviews = 76
        User 15: Train reviews = 68, Test reviews = 67
        User 17: Train reviews = 53, Test reviews = 52
        """, language='python')
        

    elif page == "Recommendation Abstract":
        st.header("Recommendation Abstract")
        
        # Display the flowchart image
        st.image('diagram.png', caption='Process Flow of Recommendation System', use_column_width=True)
        
        st.write("""
        ### Overview
        Our recommendation engine is designed to identify patterns and groupings within user data to recommend products or suggest suitable users for products. This is accomplished through the implementation of collaborative filtering methods.

        ### Types of Recommendations
        **1. Consumer/customer recommendation (user rec):**
        This type of recommendation is employed when we have one user and want to recommend multiple products. It's prevalent in customer-centric domains like entertainment, FMCG, retail, and social media, where the product range is suitable for all users.

        **2. Product recommendation (service rec):**
        This approach is used when we have one product and aim to recommend it to multiple users. It's typical in product-centric domains like banking, finance, investment, and insurance, where products are targeted to users meeting specific criteria.
        
        ### Collaborative Filtering
        Collaborative filtering serves as the foundation of our recommendation engine, leveraging user interactions and behaviors to suggest products or services. It's a technique that filters out items that a user might like based on reactions by similar users.
        
        ### Modeling Techniques
        Within collaborative filtering, we explored two distinct techniques:
        
        **User-based filtering:** This method suggests products by identifying similar users based on their preferences and past behavior.
        
        **Item-based filtering:** Contrary to user-based filtering, this technique makes recommendations based on the similarity between items rather than users.
        
        The distinction between user-based and item-based filtering lies in the focus of comparison: user-based compares the behaviors of users, while item-based compares the characteristics of the items themselves.
        
        ### Choice of Model
        After thorough analysis, we chose the ALS (Alternating Least Squares) model for our recommendation engine due to its matrix factorization technique that efficiently handles large, sparse datasets and incorporates implicit feedback.
        
        The benefits of the ALS model include scalability and more accurate predictions due to its regularization-based approach, as opposed to the distance-based approach of the KNN model.
        
        ### Testing and Conclusions
        Our testing involved cross-model comparison on various performance metrics. The ALS model outperformed the KNN model with the following RMSE scores:
        
        - **ALS RMSE**: 0.9191026086542822
        - **KNN RMSE**: 1.7873672429944476
        
        The lower RMSE score of the ALS model indicates superior performance, leading us to select it for our final recommendation system.
        
        In summary, the ALS model's ability to deliver better predictions and performance, coupled with its suitability for our sparse dataset, established it as the optimal choice for our recommendation engine.
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

    elif page == "Extra All Predictions":
        st.header("All Predictions")
        st.write("Displaying predictions for the entire test dataset.")
        
        # Display the pred_pandas DataFrame
        st.dataframe(pred_pandas)

        # total lenght of the dataframe
        st.write(f"Total length of the Test Dataframe: {pred_pandas.shape[0]}")
        
        # Calculate RMSE for the entire dataset
        rmse = sqrt(mean_squared_error(pred_pandas['rating'], pred_pandas['ALS_predicted']))
        st.write(f"Root Mean Squared Error (RMSE) for the entire dataset: {rmse:.4f}")
        
        # Additional statistics (you can add any other statistics you find relevant)
        mean_absolute_error = np.mean(np.abs(pred_pandas['rating'] - pred_pandas['ALS_predicted']))
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error:.4f}")

        # Display the number of significant errors
        significant_error_threshold = 1.5
        significant_errors = pred_pandas[np.abs(pred_pandas['rating'] - pred_pandas['ALS_predicted']) > significant_error_threshold]
        st.write(f"Number of significant errors (error > {significant_error_threshold}): {significant_errors.shape[0]}")

        # Display the number of false positives and false negatives
        false_positives = significant_errors[significant_errors['ALS_predicted'] > significant_errors['rating']].shape[0]
        false_negatives = significant_errors[significant_errors['ALS_predicted'] < significant_errors['rating']].shape[0]
        st.write(f"Number of false positives: {false_positives}")
        st.write(f"Number of false negatives: {false_negatives}")

        # Calculate True Positives (TP): Predictions within the significant error threshold
        true_positives = pred_pandas[np.abs(pred_pandas['rating'] - pred_pandas['ALS_predicted']) <= significant_error_threshold].shape[0]
        st.write(f"Number of true positives: {true_positives}")
        total_predictions = pred_pandas.shape[0]
        accuracy = true_positives / total_predictions if total_predictions else 0
        st.write(f"Accuracy: {accuracy:.2%}")






# Add a function to handle user logout
def logout_user():
    if 'logged_in' in st.session_state:
        del st.session_state['logged_in']
    if 'username' in st.session_state:
        del st.session_state['username']
    st.experimental_rerun()

# Modify the conditional logic to check for user login state and display username and logout option if logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    activity = st.sidebar.selectbox("Activity", ["Login", "Register"])
    if activity == "Login":
        login_user()
    elif activity == "Register":
        registration_form()
else:
    # Display username and logout option in the sidebar
    st.sidebar.write(f"Logged in as: {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        logout_user()

    # Call main_app function to display the main content of the app
    main_app()
