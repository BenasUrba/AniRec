# Instructions:
# Run the code, you will be prompted for a MyAnimeList username entry, find one from MyAnimeList or use my MyAnimeList
# username 'Benxsu', then wait for the recommendations to appear

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import requests
import time

# Function used to retrieve a users anime list using the MyAnimeList API
def get_user_anime_list(username, access_token):
    url = f'https://api.myanimelist.net/v2/users/{username}/animelist?fields=list_status{{score}}'
    headers = {'Authorization': f'Bearer {access_token}'}
    params = {'nsfw': 'true', 'limit': 1000, 'sort': 'list_score'}
    response = requests.get(url, headers=headers,params=params)

    if response.status_code == 200:
        anime_list = response.json()
        return anime_list
    else:
        print(f"Failed to retrieve anime list for {username}")
        return None


# Access token used for MyAnimeList API
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIn0.eyJhdWQiOiI1MDI0ODE5MGQ5ZmU4Y2RhY2UyZmYxZjFiZmEyMDhjMiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIiwiaWF0IjoxNzEzNzg3OTU2LCJuYmYiOjE3MTM3ODc5NTYsImV4cCI6MTcxNjM3OTk1Niwic3ViIjoiOTk1MjE2MCIsInNjb3BlcyI6W119.aGFfQx43UHxSu3BQcLNZNYG8i7EOuolEnIY_2U7cYNvMrAQRbJRsKy1NnPLFwNVNM1_NLhCQsb3Wu6Ef_vDPA50GtLOdKOf-MUdJ-mLMmpUJvpXTizFh4uIznh5yhbWXZtZLgqDiPVz5aMwAuXyEBPFrG6dKgVuR6wdvbQ9kWWv3gounwYgXiAPtpRIXU7U75KChHYCc8aOdWAJHZ8vZmM0C8LzP90nCcaiarcvzojnn-LuYz59ImRbzER4snznkgb35Yf0Sy2BZLmPSf6nRaEKbIyEtkuYLejcHYmj0Qyu6mMg1Y2EN0jBHsK86GDKLCEF5vw2SHLgf3_6z41oaBg"
username = input("Please enter your MyAnimeList Username: ")
anime_list = get_user_anime_list(username, access_token)
anime_ids = [id['node']['id'] for id in anime_list['data']]

start = time.time()

df = pd.read_csv('Datasets/updated_anime_data21.csv')


numerical_feats = ['Score', 'Episodes']
categorical_feats = ['Type', 'Studios', 'Source', 'Creator']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_feats),
    ('cat', OneHotEncoder(), categorical_feats),
    ('tfidf', TfidfVectorizer(stop_words='english'), 'Genres')
])

preprocessed_X = preprocessor.fit_transform(df)

cosine_similarity = cosine_similarity(preprocessed_X, preprocessed_X)

# Content recommender which takes in a list of anime IDs that the user has seen and returns the recommendations
# with the highest similarity
def content_recommender(id_list, df, cosine_similarity=cosine_similarity, top_n=50):
    total_sim_score = np.zeros(len(df))
    df_ids = df['MAL_ID']
    for mal_id in id_list:
        if mal_id in df_ids.values:
            idx = df[df['MAL_ID'] == mal_id].index[0]
            total_sim_score += cosine_similarity[idx]

    total_sim_score[np.isin(df_ids, id_list)] = 0

    min_score = np.min(total_sim_score)
    max_score = np.max(total_sim_score)
    normalized_scores = (total_sim_score - min_score) / (max_score - min_score)

    top_indices = normalized_scores.argsort()[::-1][:top_n]

    estimated_ratings = normalized_scores[top_indices]
    print("Estimated ratings:", estimated_ratings)
    average_rating = sum(estimated_ratings) / len(estimated_ratings)
    print("Average rating:", average_rating)

    return df.iloc[top_indices]


recommendations = content_recommender(anime_ids, df)
for indx, row in recommendations.iterrows():
    print(f"ID: {row['MAL_ID']} Name: {row['Name']}")

print("Time Taken: ", time.time() - start)
