# Instructions:
# Uncomment which ever evaluation you need, first code displays RMSE and MAE, second code is the hyperparameter tuning
# Third code is the precision and recall @ k, and the final code displays novelty and diversity.

# ------------------------------------------------------------------
# 1. RMSE & MAE
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
# svd_model = joblib.load('Models/SVDModel1.pkl')
# svd_model.fit(data.build_full_trainset())
#
# predictions = svd_model.test(data.build_full_trainset().build_testset())
#
# Calculates the RMSE and MAE for the predictions
# accuracy.rmse(predictions)
# accuracy.mae(predictions)
# end_time = time.time()
# print("Time took:", end_time-start_time)
#------------------------ ----------------------------------------
# # 2. Grid Search
## Used to perform Hyperparameter tuning in getting the best SVD model
# import pandas as pd
# from surprise import SVD, Dataset, Reader, accuracy
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
#     'n_factors': [100, 150, 200, 300, 500, 750, 1000, 1500, 2500],
#     'n_epochs': [10, 15, 20, 30, 50],
#     'lr_bi': [0.001, 0.005, 0.01, 0.1],
#     'reg_bi': [0.05, 0.1, 0.005, 0.001],
#     'random_state': [0],
# }
#
# svd = SVD()
#
# grid_search = GridSearchCV(SVD, param_grid=param_grid, measures=['rmse'], cv=5, n_jobs=-1, joblib_verbose=2, refit=True)
# grid_search.fit(data)
#
# best_estimator = grid_search.best_estimator['rmse']
#
# joblib.dump(best_estimator, 'SVDModel.pkl')
#
# best_params = grid_search.best_params
#
# best_svd_model = SVD(n_factors=best_params.get('n_factors', 100),
#                      n_epochs=best_params.get('n_epochs', 20),
#                      lr_all=best_params.get('lr_all', 0.01),
#                      reg_all=best_params.get('reg_all', 0.02))
#
# trainset = data.build_full_trainset()
# best_svd_model.fit(trainset)
#
#
# predictions = best_svd_model.test(trainset.build_testset())
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
## 3. Calculating Precision, Recall and F1 at K
# from collections import defaultdict
# from surprise import Dataset, Reader
# from surprise.model_selection import KFold
# import pandas as pd
# import joblib
#
## Calculates precision and recall @ k given predictions, measures it by checking if the estimated ratings are over the
## threshold, if recommendations are over the threshold, then it indicates relevant items.
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
#
# algo = joblib.load('Models/SVDModel1.pkl')
#
# avg_pre_trained = []
# avg_rec_trained = []
#
# for trainset, testset in kf.split(data):
#     predictions = algo.test(testset)
#     precisions, recalls = precision_recall_at_k(predictions, k=50, threshold=8)
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

# A pretrained model is loaded and trained on the new data which includes the new users data
svd_model = joblib.load('Models/SVDModel1.pkl')
svd_model.fit(data.build_full_trainset())

# SVD recommender which takes in a list of anime IDs that a user has seen, predicted ratings for unseen anime are
# generated based on the previous user interactions, the highest rated predicted rating anime are then recommended
def svd_recommender(seen_anime, user_id):

    anime_ids = df['anime_id'].unique()

    unrated_animes = set(anime_ids) - set(seen_anime)

    predictions = [svd_model.predict(user_id, anime_id) for anime_id in unrated_animes]

    predictions.sort(key=lambda x: x.est, reverse=True)

    return predictions
#
# # -------------- V Calculates Diversity of Recommendations V ----------------
#
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
    user_recs = svd_recommender(user_inter, user_id)
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
#     recommendations = svd_recommender(user_iter, user)
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
