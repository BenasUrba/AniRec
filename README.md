# AniRec Personalized Anime Recommendation Website

![AniRecLogo3](https://github.com/BenasUrba/AniRec/assets/141029588/ab456cda-796b-4b35-bb73-fc9fc995a65c)

Recommender systems are widely used in different online platforms such as e-commerce websites and online streaming services. These systems are used to help find and suggest relevant items to users, often based on the user’s past interactions which can help personalize the recommendations. Another benefit of recommender systems is its ability to save the time taken for users to find recommendations manually. 

Anime are Japanese cartoons that have been rising in popularity to overseas consumers in recent years. The anime industry reached a revenue of 2.93 trillion Japanese Yen (approximately £18.11 billion GBP) in 2022 (Statista Research Department, 2024), where 1.46 trillion Japanese Yen (approximately £9.03 billion GBP) make up the overseas sales of anime (Statista Research Department, 2024). Rapid growth for the market, means more animes will be produced as there are more users to satisfy and appeal to. With vast amount of animes currently available and releasing, there comes an issue of finding relevant animes that users will enjoy, here recommendation systems can help users finding relevant recommendations.

To address this challenge, this paper focuses on improving the performance of anime recommender systems by utilizing various techniques and methodologies, such as extensive data preprocessing and hyperparameter tuning. Collaborative filtering (CF), content-based filtering (CB) and hybrid methods are among the three most popular methods used for recommender systems. The proposed model in this paper is a hybrid model which leverages collaborative Singular Value Decomposition (SVD) model, a matrix factorization technique which analyses user-item interactions to find similar users and items. A content-based model which explores past traditional features such as anime genres, and incorporates more novel attributes like creator values, such as writers or directors. K-nearest neighbour (KNN) is also introduced as a base model for evaluations comparisons. Finally, the main model proposed in this paper is the combination of the SVD and content-based approaches to surpass popular issues such as Cold Start problem and provide more accurate and personalized recommendations. These models will be evaluated on various metrics such as precision @ k, diversity and novelty to demonstrate the performance of the models in more depth. To ensure personalization, users will be able to input their MyAnimeList username to generate recommendations based of their previous interactions.


AniRec can offer personalized recommendations based on a users previous anime interactions from their accounts in MyAnimeList using a hybrid system of Singular Value Decomposition and a Content-Based Model. Moreover, if users do not have a MyAnimeList account or do not want personalization in their recommendations and instead want similar anime recommendations, they can opt for a single anime id option, which displays anime which have the most similar features to the one inputted. As this focuses on the features of the anime, a content based model was utilized for its capabilities at identifying similarities between features in anime. It also offers other features such as the ability to see popular anime at the home page for fast recommendations or if users dont know for which anime they want recomendations. Furtermore, each anime poster is interactable and redirects the user to a MyAnimeList website of that anime. Once recommendations are generated, 3 recommendations are shown at once, however, a user can use the 'Next' and 'Previous' button to access the next 3 or previous 3 anime, there is a total of 52 anime.

This website utilizes the MyAnimeList API and therefore requires an access token to use, this has been removed from the code due to confidentiality purposes, but the code for receiving and using the access token has been left if anyone wishes to try out the website/code. Moreover, the pre-trained Singular Value Decomposition (SVD) model has not been added as it was too big of a file size (52MB). 

# AniRec Homepage
![AniRec_Home](https://github.com/BenasUrba/AniRec/assets/141029588/a607f4ef-b8f2-4f40-8b82-38bc124b68c0)

# AniRec Generated Recommendations Page
![AniRec_Recs](https://github.com/BenasUrba/AniRec/assets/141029588/b20f7170-b8c8-412d-8427-ebd1956a6611)

# AniRec website redirect once you click on poster (Example: Fullmetal Alchemist: Brotherhood)
![image](https://github.com/BenasUrba/AniRec/assets/141029588/5344693f-efb3-4c12-8587-c382cdf810a4)


# AniRec Logo
![AniRecLogo3](https://github.com/BenasUrba/AniRec/assets/141029588/ab456cda-796b-4b35-bb73-fc9fc995a65c)



