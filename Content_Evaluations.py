# Instructions:
# Can calculate Diversity, Novelty and Precision. Comment in/out whichever you want to see

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
import numpy as np

df = pd.read_csv('Datasets/updated_anime_data21.csv')


numerical_feats = ['Score', 'Episodes']
categorical_feats = ['Type', 'Creator', 'Studios', 'Source']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_feats),
    ('cat', OneHotEncoder(), categorical_feats),
    ('tfidf', TfidfVectorizer(stop_words='english'), 'Genres')
])

preprocessed_X = preprocessor.fit_transform(df)


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

    top_indices = normalized_scores.argsort()[::-1][:50]

    return df.iloc[top_indices]

users_df = pd.read_csv('Datasets/100+Users5+Anime2.csv')

# ------------------ V Calculates the Diversity of the Recommendations V ------------------

all_genres = set()

genre_string = ','.join(df['Genres'])

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
def calculate_diversity(recommendations):
    total_dissimilarities = []
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            genres1 = recommendations.iloc[i]['Genres'].split(', ')
            genres2 = recommendations.iloc[j]['Genres'].split(', ')
            dissimilarity = 1 - compute_genre_similarity(genres1, genres2)
            total_dissimilarities.append(dissimilarity)

    diversity_score = np.mean(total_dissimilarities)
    return diversity_score

total_diversities = []

for user_id in users_df['user_id'].unique()[:10]:
    user_inter = users_df[users_df['user_id'] == user_id]['anime_id'].tolist()
    user_recs = content_recommender(user_inter, df)
    diversities = calculate_diversity(user_recs)
    total_diversities.append(diversities)

# The overall diversity of the models performance is calculated
div_score = np.mean(total_diversities)
print(f"Diversity Score: {div_score}")

# ---------------- V Calculates the Novelty of the Recommendations V -----------------

# members_df = pd.read_csv('members.csv')
#
# # Members values of each anime are normalized 0-1 to allow for accurate and consistent comparisons against the SVD model
# scaler = MinMaxScaler()
# members_df['members_normalized'] = scaler.fit_transform(members_df[['Members']])
#
# novelty_scores = []
#
# # Measures the popularity of each recommendation by getting the mean score of the popularity (member values) of each
## recommendations
# for user in users_df['user_id'].unique():
#     user_novelty = []
#     user_iter = users_df[users_df['user_id'] == user]['anime_id'].tolist()
#     recommendations = content_recommender(user_iter, df)
#     for indx, rec in recommendations.iterrows():
#         pop_score = members_df[members_df['MAL_ID'] == rec['MAL_ID']]['members_normalized'].values
#         user_novelty.append(pop_score[0] if len(pop_score) > 0 else 0)
#     mean_novelty_score = np.mean(user_novelty)
#     novelty_scores.append(mean_novelty_score)
#
## Mean novelty is calculated to get the overall novelty performance of the model
# overall_novelty = np.mean(novelty_scores)
# print(f"Mean Novelty Score: {overall_novelty}")

# ---------------- V Calculates (MAP) Precision @ k V ------------------------
#
## Precision @ k is calculated by measuring how many relevant (anime that the user has already interacted with) are in
## the recommendations
# def precision_at_k(users_df, k):
#     user_precisions = []
#     for user_id in users_df['user_id'].unique():
#         user_interactions = users_df[users_df['user_id'] == user_id]['anime_id'].tolist()
#         user_recs = content_recommender(user_interactions, df, precision=True)
#         user_recs_list = user_recs['MAL_ID'].tolist()
#         relevant_count = sum(anime_id in user_interactions for anime_id in user_recs_list[:k])
#         user_precision = relevant_count / k if k > 0 else 0
#         user_precisions.append(user_precision)
#     mean_precision = sum(user_precisions) / len(user_precisions) if user_precisions else 0
#     return mean_precision
#
# # Mean precision is calculated to measure the overall precision performance of the model
# mean_precision = precision_at_k(users_df, 50)
# print("Mean Average Precision at 10:", mean_precision)
