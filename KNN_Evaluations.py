# Instructions:
# Uncomment which ever evaluation you need, first code displays RMSE and MAE, second code is the hyperparameter tuning
# Third code is the precision and recall @ k, and the final code displays novelty and diversity.

# --------------------------------------------------------------
## 1. RMSE & MAE
# import pandas as pd
# from surprise import Dataset, Reader
# from surprise import accuracy
# import time
# import joblib
#
# start_time = time.time()
#
# df = pd.read_csv('Datasets/100+Users5+Anime2.csv')
#
# reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))
#
# data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader=reader)
#
# knn_model = joblib.load('Models/KNNMModel.pkl')
# knn_model.fit(data.build_full_trainset())
#
# predictions = knn_model.test(data.build_full_trainset().build_testset())
#
## RMSE and MAE are calculated based on the prediction
# accuracy.rmse(predictions)
# accuracy.mae(predictions)
# end_time = time.time()
# print("Time took:", end_time-start_time)
#------------------------ ----------------------------------------
# # 2. Grid Search
## Used to perform Hyperparameter tuning in getting the best KNNWIthMeans model
# import pandas as pd
# from surprise import KNNWithMeans, Dataset, Reader, accuracy
# from surprise.model_selection import GridSearchCV
# import time
# import joblib
#
# start_time = time.time()
# df = pd.read_csv('Datasets/100+Users5+Anime2.csv')
#
# reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))
#
# data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader=reader)
#
# param_grid = {
#     'k': [20, 50, 100, 200],
#     'min_k': [1, 3, 5, 10],
#     'sim_options': {'name': ['pearson', 'msd'],
#                     'user_based': [True, False]},
#     'random_state': [0],
# }
#
# grid_search = GridSearchCV(KNNWithMeans, param_grid=param_grid, measures=['rmse'], cv=5, n_jobs=-1, joblib_verbose=2, refit=True)
# grid_search.fit(data)
#
# best_estimator = grid_search.best_estimator['rmse']
#
# joblib.dump(best_estimator, 'KNNModel.pkl')
#
# best_params = grid_search.best_params
#
# best_knn_model = KNNWithMeans(k=best_params.get('k', 20),
#                      min_k=best_params.get('min_k', 5),
#                      sim_options=best_params.get('sim_options', {'name': 'pearson', 'user_based': True}),
#                      random_state=best_params.get('random_state', 0))
#
# trainset = data.build_full_trainset()
# best_knn_model.fit(trainset)
#
#
# predictions = best_knn_model.test(trainset.build_testset())
# best_rmse = accuracy.rmse(predictions)
# best_mae = accuracy.mae(predictions)
#
# end_time = time.time()
#
# print(f"Best Parameters: {best_params}")
# print(f"Best RMSE: {best_rmse}")
# print(f"Best MAE: {best_mae}")
# print(f"Time taken: {end_time-start_time}")

# ------------------------------------------------
# # 3. Calculating Precision and Recall  at k
# from collections import defaultdict
# from surprise import Dataset, Reader
# from surprise.model_selection import KFold
# import pandas as pd
# import joblib
#
## Calculates precision and recall @ k given predictions, measures it by checking if the estimated ratings are over the
## threshold, if recommendatoins are over the threshold, then it indicates relevant items.
# def precision_recall_at_k(predictions, k=10, threshold=8):
#
#     user_est_true = defaultdict(list)
#     for uid, _, true_r, est, _ in predictions:
#         user_est_true[uid].append((est, true_r))
#
#     precisions = dict()
#     recalls = dict()
#     for uid, user_ratings in user_est_true.items():
#
#         user_ratings.sort(key=lambda x: x[0], reverse=True)
#
#         n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
#
#         n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
#
#         n_rel_and_rec_k = sum(
#             ((true_r >= threshold) and (est >= threshold))
#             for (est, true_r) in user_ratings[:k]
#         )
#
#         precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
#
#         recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0
#
#     return precisions, recalls
#
#
# df = pd.read_csv('Datasets/100+Users5+Anime2.csv')
#
# reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))
#
# data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader=reader)
#
#
# kf = KFold(n_splits=5, random_state=42)
# algo = joblib.load('Models/KNNMModel.pkl')
#
# avg_pre_trained = []
# avg_rec_trained = []
#
# for trainset, testset in kf.split(data):
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=10, threshold=8)
#     avg_pre_trained.append(sum(prec for prec in precisions.values()) / len(precisions))
#     avg_rec_trained.append(sum(rec for rec in recalls.values()) / len(recalls))
#
#
# avg_pre_trained = sum(avg_pre_trained) / len(avg_pre_trained)
# avg_rec_trained = sum(avg_rec_trained) / len(avg_rec_trained)
#
# print("Average Precision k=10 : {:.4f}".format(avg_pre_trained))
# print("Average Recall k=10 : {:.4f}".format(avg_rec_trained))

# --------------------------------------------------------
# # 4. Calculates the Novelty and Diversity for the model
import pandas as pd
from surprise import Dataset, Reader
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import time
import joblib
import numpy as np

df = pd.read_csv('Datasets/100+Users5+Anime2.csv')

reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 10))

data = Dataset.load_from_df(df[['user_id', 'anime_id', 'rating']], reader=reader)

start = time.time()

# A pretrained KNNWithMeans model is loaded
knn_model = joblib.load('Models/KNNMModel.pkl')


# KNN recommender which takes in a list of anime IDs that a user has seen, predicted ratings for unseen anime are
# generated based on the previous user interactions, the highest rated predicted rating anime are then recommended
def knn_recommender(seen_anime, user_id):

    anime_ids = df['anime_id'].unique()

    unrated_animes = set(anime_ids) - set(seen_anime)

    predictions = [knn_model.predict(user_id, anime_id) for anime_id in unrated_animes]

    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions
#
# # -------------- V Calculates Diversity of Recommendations V ----------------
# #
genre_df = pd.read_csv('Datasets/updated_anime_data21.csv')

all_genres = set()

genre_string = ','.join(genre_df['Genres'])

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
def calculate_diversity(recommended_item_ids, genre_df):
    total_dissimilarities = []

    for i in range(len(recommended_item_ids)):
        for j in range(i + 1, len(recommended_item_ids)):
            item_id1 = recommended_item_ids[i].iid
            item_id2 = recommended_item_ids[j].iid

            if item_id1 in genre_df['MAL_ID'].values and item_id2 in genre_df['MAL_ID'].values:

                genres1 = genre_df.loc[genre_df['MAL_ID'] == item_id1, 'Genres'].iloc[0].split(', ')
                genres2 = genre_df.loc[genre_df['MAL_ID'] == item_id2, 'Genres'].iloc[0].split(', ')

                dissimilarity = 1 - compute_genre_similarity(genres1, genres2)
                total_dissimilarities.append(dissimilarity)

    diversity_score = np.mean(total_dissimilarities)
    return diversity_score

total_diversities = []

for user_id in df['user_id'].unique():
    user_inter = df[df['user_id'] == user_id]['anime_id'].tolist()
    user_recs = knn_recommender(user_inter, user_id)
    diversities = calculate_diversity(user_recs, genre_df)
    total_diversities.append(diversities)

# The overall diversity of the models performance is calculated
div_score = np.mean(total_diversities)
end = time.time()
print(f"Diversity Score: {div_score}")
#
# # ---------- V Calculates the Novelty of recommendations V --------------
# #
# item_popularity = df['anime_id'].value_counts().reset_index()
# item_popularity.columns = ['anime_id', 'popularity']
#
# # Popularity values of each anime are normalized 0-1 to allow for accurate and consistent comparisons against the SVD model
# scaler = MinMaxScaler()
# item_popularity['popularity_normalized'] = scaler.fit_transform(item_popularity[['popularity']])
#
# novelty_scores = []
#
# for user in df['user_id'].unique():
#     user_novelty = []
#     user_iter = df[df['user_id'] == user]['anime_id'].tolist()
#     recommendations = knn_recommender(user_iter, user)
#     recs = (rec.iid for rec in recommendations)
#     for rec in recs:
#         pop_score = item_popularity[item_popularity['anime_id'] == rec]['popularity_normalized'].values
#         user_novelty.append(pop_score[0] if len(pop_score) > 0 else 0)
#     mean_novelty_score = np.mean(user_novelty)
#     novelty_scores.append(mean_novelty_score)
#
## Mean novelty is calculated to get the overall novelty performance of the model
# overall_novelty = np.mean(novelty_scores)
# print(f"Mean Novelty Score: {overall_novelty}")
