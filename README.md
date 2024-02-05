# Tens ML Coding Challenge Movie Recommendation Algorithm
 Tens ML Coding Challenge Movie Recommendation Algorithm

# Streamlit Movie Recommendation App

## Overview
This project is a Streamlit-based web application that recommends movies to users based on their preferences and previous ratings. It leverages a collaborative filtering approach using the Alternating Least Squares (ALS) algorithm to predict user ratings for movies they haven't seen yet. The app is designed to demonstrate the capabilities of a simple movie recommendation algorithm using a dataset split into training and testing sets to ensure accurate and personalized recommendations.

## Features
- **Data Overview:** A snapshot of the raw data used for the recommendations.
- **Feature Enhancement:** Details of data transformations or new columns introduced to aid the recommendation engine.
- **Test Train Split Overview:** An explanation of the methodology used to split the dataset into training and testing datasets.
- **Recommendation Abstract:** Rationale behind the model selection and insights from testing the algorithm.
- **Recommendation Demo:** A feature that allows users to input a user ID to see recommended movies and the predicted ratings, as well as statistics on the model's performance based on the test dataset.

## Project Structure
- `app.py`: The main Streamlit application script containing the login system, data processing, model training, and recommendation logic.
- 'TENS_ML_Challenge.ipynb': Jupyter notebook containing the code for the project.
- `requirements.txt`: A list of project dependencies.
- `data/`: Directory containing the dataset files (movies.csv, ratings.csv).

## Getting Started

### Prerequisites
- Python 3.6 or later
- Pip

### Installation
1. Clone the repository: git clone https://github.com/sree369nidhi/Tens-ML-Coding-Challenge-Movie-Recommendation-Algorithm.git

2. Navigate to the project directory: cd Tens-ML-Coding-Challenge-Movie-Recommendation-Algorithm

3. Install the required dependencies: pip install -r requirements.txt

4. Run the Streamlit app: streamlit run app.py

5. Open a web browser and navigate to http://localhost:8501 to access the app.

## Acknowledgments
- Dataset provided by [Kaggle](https://www.kaggle.com/).
- Streamlit for making data app creation a breeze.

