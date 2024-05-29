# Instructions:
# Can calculate Diversity, Novelty and Precision. Comment in/out whichever you want to see

import pandas as pd
from surprise import Dataset, Reader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import time
import joblib

svd_df = pd.read_csv('Datasets/100+Users5+Anime2.csv')

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))

data = Dataset.load_from_df(svd_df[['user_id', 'anime_id', 'rating']], reader=reader)

# A Pretrained SVD model is loaded
svd_model = joblib.load('Models/SVDModel1.pkl')

# SVD recommender which takes in a list of seen anime IDs and returns the recommendations sorted by the highest predicted rating
def svd_recommender(seen_anime, user_id, precision=False):

    anime_ids = svd_df['anime_id'].unique()

    if not precision:
        anime_ids = set(anime_ids) - set(seen_anime)

    predictions = [svd_model.predict(user_id, anime_id) for anime_id in anime_ids]

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


cosine_similarity2 = cosine_similarity(preprocessed_X, preprocessed_X)

# Content recommender which takes in a list of anime IDs that the user has seen and returns the recommendations
# with the highest similarity
def content_recommender(id_list, df, cosine_similarity=cosine_similarity2, precision=False):
    total_sim_score = np.zeros(len(df))
    df_ids = df['MAL_ID']
    for mal_id in id_list:
        if mal_id in df_ids.values:
            idx = df[df['MAL_ID'] == mal_id].index[0]
            total_sim_score += cosine_similarity[idx]

    if not precision:
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
    normalized_svd_scores = {iid: score / 10.0 for iid, score in svd_preds.items()}

    content_ratings_dic = dict(zip(content_rec, content_ratings))

    svd_weights = 0.5

    hybrid_scores = {iid: (svd_weights * normalized_svd_scores.get(iid, 0)) + ((1 - svd_weights) * content_ratings_dic.get(iid, 0)) for iid in set(normalized_svd_scores) | set(content_ratings_dic)}

    sorted_scores = sorted(hybrid_scores.items(), key=lambda item : item[1], reverse=True)

    top_recommendations = [iid for iid, _ in sorted_scores[:50]]

    return top_recommendations

# ------------------- V Diversity for Recommendations V --------------------

all_genres = set()

genre_string = ','.join(content_df['Genres'])

all_individual_genres = genre_string.split(',')

all_genres.update(all_individual_genres)

# Computes the similarity between two lists of genres
def compute_genre_similarity(genres1, genres2):
    binary1 = np.array([int(genre in genres1) for genre in all_genres])
    binary2 = np.array([int(genre in genres2) for genre in all_genres])
    similarity = cosine_similarity([binary1], [binary2])[0][0]
    return similarity

# Calculates the diversity of the recommendations by taking two anime at a time, then calculating the dissimilarity
# between the two anime's genres.
def calculate_diversity(list_of_ids, dataset):
    total_dissimilarities = []
    for i in range(len(list_of_ids)):
        for j in range(i + 1, len(list_of_ids)):
            try:
                genres1 = dataset[dataset['MAL_ID'] == list_of_ids[i]]['Genres'].iloc[0].split(', ')
                genres2 = dataset[dataset['MAL_ID'] == list_of_ids[j]]['Genres'].iloc[0].split(', ')
                dissimilarity = 1 - compute_genre_similarity(genres1, genres2)
                total_dissimilarities.append(dissimilarity)
            except:
                continue

    diversity_score = np.mean(total_dissimilarities)
    return diversity_score

total_diversities = []

start = time.time()
for user_id in svd_df['user_id'].unique():
    user_iter = svd_df[svd_df['user_id'] == user_id]['anime_id'].tolist()
    svd_recommendations = svd_recommender(user_iter, user_id)
    content_recommendations, content_ratings = content_recommender(user_iter, content_df)
    recommendations = hybrid_recommender(svd_recommendations, content_recommendations, content_ratings)
    diversities = calculate_diversity(recommendations, content_df)
    total_diversities.append(diversities)

# The overall diversity of the models performance is calculated
div_score = np.mean(total_diversities)
end = time.time()
print(f"Diversity Score: {div_score}")
print(f"Time taken: {end-start}")

# --------------------------- V Novelty for Recommendations V --------------------
#
# members_df = pd.read_csv('members.csv')
#
# # Members values of each anime are normalized 0-1 to allow for accurate and consistent comparisons against the SVD model
# scaler = MinMaxScaler()
# members_df['members_normalized'] = scaler.fit_transform(members_df[['Members']])
#
# novelty_scores = []
#
#
# # Measures the popularity of each recommendation by getting the mean score of the popularity (member values) of each
# for user in svd_df['user_id'].unique():
#     user_novelty = []
#     user_iter = svd_df[svd_df['user_id'] == user]['anime_id'].tolist()
#     svd_recommendations = svd_recommender(user_iter, user)
#     content_recommendations, content_ratings = content_recommender(user_iter, content_df)
#     recommendations = hybrid_recommender(svd_recommendations, content_recommendations, content_ratings)
#     for rec in recommendations:
#         pop_score = members_df[members_df['MAL_ID'] == rec]['members_normalized'].values
#         user_novelty.append(pop_score[0] if len(pop_score) > 0 else 0)
#     mean_novelty_score = np.mean(user_novelty)
#     novelty_scores.append(mean_novelty_score)
#
## Mean novelty is calculated to get the overall novelty performance of the model
# overall_novelty = np.mean(novelty_scores)
# print(f"Mean Novelty Score: {overall_novelty}")

# ----------- V Precision for recommendations V ---------------------
#
## Precision @ k is calculated by measuring how many relevant (anime that the user has already interacted with) are in
## the recommendations
# def precision_at_k(users_df, k):
#     user_precisions = []
#     for user_id in users_df['user_id'].unique():
#         user_iter = svd_df[svd_df['user_id'] == user_id]['anime_id'].tolist()
#         svd_recommendations = svd_recommender(user_iter, user_id, precision=True)
#         content_recommendations, content_ratings = content_recommender(user_iter, content_df, precision=True)
#         recommendations = hybrid_recommender(svd_recommendations, content_recommendations, content_ratings)
#         relevant_count = sum(anime_id in user_iter for anime_id in recommendations[:k])
#         user_precision = relevant_count / k if k > 0 else 0
#         user_precisions.append(user_precision)
#     mean_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
#     return mean_precision
#
# # Mean precision is calculated to measure the overall precision performance of the model
# mean_precision = precision_at_k(svd_df, 10)
# print("Mean Average Precision at 10:", mean_precision)
# #
