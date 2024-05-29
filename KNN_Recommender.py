# Instructions:
# Run the code, you will be prompted for a MyAnimeList username entry, find one from MyAnimeList or use my MyAnimeList
# username 'Benxsu', then wait for the recommendations to appear

import pandas as pd
from surprise import Dataset, Reader
import requests
import time
import joblib

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

# Function which takes in a users anime list and retrieves the ratings that a user has given for each anime
def get_user_ratings(anime_list):
    user_id = 10000
    if anime_list is None:
        return None
    ratings = []
    seen_animes = []
    for item in anime_list['data']:
        id = item['node']['id']
        try:
            status = item['list_status']['status']
            if status not in ['completed', 'watching']:
                continue
            rating = item['list_status']['score'] if 'list_status' in item and 'score' in item['list_status'] else None
            if rating is not None:
                ratings.append((user_id, id, rating))
            seen_animes.append(id)
        except KeyError:
            continue

    return ratings, seen_animes

df = pd.read_csv('Datasets/100+Users5+Anime2.csv')


# Access token used for MyAnimeList API
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIn0.eyJhdWQiOiI1MDI0ODE5MGQ5ZmU4Y2RhY2UyZmYxZjFiZmEyMDhjMiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIiwiaWF0IjoxNzEzNzg3OTU2LCJuYmYiOjE3MTM3ODc5NTYsImV4cCI6MTcxNjM3OTk1Niwic3ViIjoiOTk1MjE2MCIsInNjb3BlcyI6W119.aGFfQx43UHxSu3BQcLNZNYG8i7EOuolEnIY_2U7cYNvMrAQRbJRsKy1NnPLFwNVNM1_NLhCQsb3Wu6Ef_vDPA50GtLOdKOf-MUdJ-mLMmpUJvpXTizFh4uIznh5yhbWXZtZLgqDiPVz5aMwAuXyEBPFrG6dKgVuR6wdvbQ9kWWv3gounwYgXiAPtpRIXU7U75KChHYCc8aOdWAJHZ8vZmM0C8LzP90nCcaiarcvzojnn-LuYz59ImRbzER4snznkgb35Yf0Sy2BZLmPSf6nRaEKbIyEtkuYLejcHYmj0Qyu6mMg1Y2EN0jBHsK86GDKLCEF5vw2SHLgf3_6z41oaBg"
username = input("Please enter your MyAnimeList Username: ")
anime_list = get_user_anime_list(username, access_token)
if anime_list is not None:
    ratings, seen_anime = get_user_ratings(anime_list)
    if ratings:
        user_df = pd.DataFrame(ratings, columns=['user_id', 'anime_id', 'rating'])
        df = pd.concat([df, user_df], ignore_index=True)
else:
    print(f"Could not retrieve information from account: {username}")
    seen_anime = []

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))

data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader=reader)

start = time.time()

# Model is loaded and trained on new data that includes the users data
knn_model = joblib.load('Models/KNNMModel.pkl')
knn_model.fit(data.build_full_trainset())

# KNN recommender which takes in a list of anime IDs that a user has seen, predicted ratings for unseen anime are
# generated based on the previous user interactions, the highest rated predicted rating anime are then recommended
def knn_recommender(seen_anime, user_id):

    anime_ids = df['anime_id'].unique()

    unrated_animes = set(anime_ids) - set(seen_anime)

    predictions = [knn_model.predict(user_id, anime_id) for anime_id in unrated_animes]

    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions


user_id = 10000
name_ids = pd.read_csv('Datasets/CF_id_name2.csv')
recommendations = knn_recommender(seen_anime, user_id)

print("Top 5 movie recommendations for user", user_id)

for recommendation in recommendations[:50]:
    title =  name_ids.loc[name_ids['anime_id'] == recommendation.iid, 'Name'].values[0]
    print(f"Movie ID: {title} - Estimated Rating: {recommendation.est}")

print("Time Taken: ", time.time() - start)