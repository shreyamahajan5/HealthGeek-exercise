import streamlit as st
import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

# Load dataset
dataset = pd.read_csv("megaGymDataset.csv")

# Drop rows with missing values in the 'Desc' column
dataset = dataset.dropna(subset=['Desc'])

# Preprocess text data
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(dataset['Desc'])

# Compute cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Function to recommend exercises with workout plan
def recommend_exercises_with_plan(user_level, body_part, num_recommendations=5):
    filtered_dataset = dataset[(dataset['Level'] == user_level) & (dataset['BodyPart'] == body_part)]
    if filtered_dataset.empty:
        return None, f"No {user_level} level exercises found for the {body_part} body part."
    
    index = filtered_dataset.index[0] if len(filtered_dataset.index) > 0 else 0
    
    if index >= len(dataset):
        return None, f"Index out of bounds error."
    
    sim_scores = list(enumerate(cosine_sim[index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    exercise_indices = [i[0] for i in sim_scores]
    recommended_exercises = dataset.iloc[exercise_indices]
    
    return recommended_exercises, None

# Streamlit web app
st.title('Exercise Recommendation System')

# Select skill level and body part
user_level = st.selectbox('Select your skill level:', ['Beginner', 'Intermediate', 'Expert'])
body_part = st.selectbox('Select the body part:', ['Abdominals', 'Quadriceps', 'Shoulders', 'Chest', 'Biceps'])
num_recommendations = st.slider('Number of recommendations', min_value=1, max_value=10, value=5)

# Button to recommend exercises
if st.button('Recommend Exercises'):
    recommended_exercises, error_message = recommend_exercises_with_plan(user_level, body_part, num_recommendations)
    
    # Display recommended exercises in a table
    if recommended_exercises is not None:
        st.subheader("Workout Plan:")
        workout_plan_data = []
        for exercise in recommended_exercises.itertuples():
            repetitions = random.randint(1, 5)  # Random repetitions between 1 and 5
            break_time = random.randint(1, 10)  # Random break time between 1 and 10
            workout_plan_data.append([
                exercise.Title,
                exercise.Desc,
                f"{repetitions} times",  # Include units for "Repeat" column
                f"{break_time} minutes"  # Include units for "Break time" column
            ])
        workout_plan_df = pd.DataFrame(workout_plan_data, columns=['Exercise', 'Description', 'Repeat', 'Break time'])

        # Display the table with text wrapping for "Repeat" and "Break time" columns
        st.table(workout_plan_df)
    else:
        st.write(error_message)
