# Instructions:
# Run the code, you will be prompted for a MyAnimeList username entry, find one from MyAnimeList or use my MyAnimeList
# username 'Benxsu', then wait for the recommendations to appear

import pandas as pd
from surprise import Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import requests
import numpy as np
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

svd_df = pd.read_csv('Datasets/100+Users5+Anime2.csv')

# Access token used for MyAnimeList API
access_token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIn0.eyJhdWQiOiI1MDI0ODE5MGQ5ZmU4Y2RhY2UyZmYxZjFiZmEyMDhjMiIsImp0aSI6IjU1YmE2ZTlhZmMxN2JiYjFhMThlYzViOTMzMzQ0ZGY0MDc5ZjJkNWU4YjdlYmRiMmY2N2E5NmVlNGM5YjFhMDE5NWVkYmUzNWJmYzI2YzNjIiwiaWF0IjoxNzEzNzg3OTU2LCJuYmYiOjE3MTM3ODc5NTYsImV4cCI6MTcxNjM3OTk1Niwic3ViIjoiOTk1MjE2MCIsInNjb3BlcyI6W119.aGFfQx43UHxSu3BQcLNZNYG8i7EOuolEnIY_2U7cYNvMrAQRbJRsKy1NnPLFwNVNM1_NLhCQsb3Wu6Ef_vDPA50GtLOdKOf-MUdJ-mLMmpUJvpXTizFh4uIznh5yhbWXZtZLgqDiPVz5aMwAuXyEBPFrG6dKgVuR6wdvbQ9kWWv3gounwYgXiAPtpRIXU7U75KChHYCc8aOdWAJHZ8vZmM0C8LzP90nCcaiarcvzojnn-LuYz59ImRbzER4snznkgb35Yf0Sy2BZLmPSf6nRaEKbIyEtkuYLejcHYmj0Qyu6mMg1Y2EN0jBHsK86GDKLCEF5vw2SHLgf3_6z41oaBg"
username = input("Please enter your MyAnimeList Username: ")
anime_list = get_user_anime_list(username, access_token)
if anime_list is not None:
    ratings, seen_anime = get_user_ratings(anime_list)
    if ratings:
        user_df = pd.DataFrame(ratings, columns=['user_id', 'anime_id', 'rating'])
        svd_df = pd.concat([svd_df, user_df], ignore_index=True)
else:
    print(f"Could not retrieve information from account: {username}")
    seen_anime = []



reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))

data = Dataset.load_from_df(svd_df[['user_id', 'anime_id', 'rating']], reader=reader)

start = time.time()

# Model is once again retrained on the data which includes the users data
svd_model = joblib.load('Models/SVDModel1.pkl')
svd_model.fit(data.build_full_trainset())

# SVD recommender which takes in a list of seen anime IDs and returns the recommendations sorted by the highest
# predicted rating
def svd_recommender(seen_anime):
    user_id = 10000

    anime_ids = svd_df['anime_id'].unique()

    unrated_animes = set(anime_ids) - set(seen_anime)

    predictions = [svd_model.predict(user_id, anime_id) for anime_id in unrated_animes]

    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions

# ------------------ ^ SVD Model ^ ----------------------------
# ------------------ V Content Based Model V ----------------------------

content_df = pd.read_csv('Datasets/updated_anime_data21.csv')

numerical_feats = ['Score', 'Episodes']
categorical_feats = ['Type', 'Studios', 'Source', 'Creator']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_feats),
    ('cat', OneHotEncoder(), categorical_feats),
    ('tfidf', TfidfVectorizer(stop_words='english'), 'Genres')
])

preprocessed_X = preprocessor.fit_transform(content_df)


cosine_similarity = cosine_similarity(preprocessed_X, preprocessed_X)

# Content recommender which takes in a list of anime IDs that the user has seen and returns the recommendations
# with the highest similarity
def content_recommender(id_list, df, cosine_similarity=cosine_similarity):
    total_sim_score = np.zeros(len(df))
    df_ids = df['MAL_ID']
    for mal_id in id_list:
        if mal_id in df_ids.values:
            idx = df[df['MAL_ID'] == mal_id].index[0]
            total_sim_score += cosine_similarity[idx]

    # seen animes scores are set to 0 to avoid recommending the animes that the user has already seen.
    total_sim_score[np.isin(df_ids, id_list)] = 0

    min_score = np.min(total_sim_score)
    max_score = np.max(total_sim_score)
    normalized_scores = (total_sim_score - min_score) / (max_score - min_score)

    top_indices = normalized_scores.argsort()[::-1]

    estimated_ratings = normalized_scores[top_indices]

    return df.iloc[top_indices]['MAL_ID'], estimated_ratings

# ---------------------- V Hybrid Model V ----------------------------

# Hybrid recommender which takes the recommendations from the SVD and Content-based models and the content ratings
# (cosine similarity scores), then multiplies the content ratings by 10 to get a 0-10 range, these scores are then added
# to the SVD predicted scores and divided by 2 to get the overall score, the highest recommendations are then recommended.
def hybrid_recommender(svd_recs, content_rec, content_ratings):

    svd_preds = {pred.iid: pred.est for pred in svd_recs}

    normalized_content_scores = [rating * 10 for rating in content_ratings]

    content_ratings_dic = dict(zip(content_rec, normalized_content_scores))

    svd_weights = 0.5

    hybrid_scores = {iid: (svd_weights * svd_preds.get(iid, 0)) + ((1 - svd_weights) * content_ratings_dic.get(iid, 0)) for iid in set(svd_preds) | set(content_ratings_dic)}

    sorted_scores = sorted(hybrid_scores.items(), key=lambda item : item[1], reverse=True)

    top_recommendations = [iid for iid, _ in sorted_scores[:50]]

    return top_recommendations


svd_recommendations = svd_recommender(seen_anime)

content_recommendations, content_ratings = content_recommender(seen_anime, content_df)

recommendations = hybrid_recommender(svd_recommendations, content_recommendations, content_ratings)


for recommendation in recommendations:
    title = content_df[content_df['MAL_ID'] == recommendation]['Name'].values
    print(f"Title: {title}")

end = time.time()

print("Time Taken:", end - start)

