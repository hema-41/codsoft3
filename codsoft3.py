import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
#Sample data
data = { 
    'user_id':[1,1,1,2,2,3,3,4,4],
    'movie_id':[1,2,3,1,4,2,3,1,4],
    'rating':[5,4,1,4,5,5,2,3,4]
}
#Convert to DataFrame
df  = pd.DataFrame(data)
user_item_matrix = df.pivot_table(index='user_id',columns='movie_id',values='rating')
user_item_matrix.fillna(0, inplace=True)
print(user_item_matrix)
user_similarity=cosine_similarity(user_item_matrix)
print(user_similarity)
def predict_ratings(user_item_matrix, user_similarity):
    mean_user_rating = user_item_matrix.mean(axis=1)
    ratings_diff = (user_item_matrix - mean_user_rating[:,np.newaxis])
    pred = mean_user_rating[:,np.newaxis]+user_similarity.dot(ratings_diff)/ np.array([np.abs(user_similarity).sum(axis=1)])
    return pred

predicted_ratings = predict_ratings(user_item_matrix.values,user_similarity)
predicted_ratings_df = pd.DataFrame(predicted_ratings, index=user_item_matrix.index, columns=user_item_matrix.columns)
print(predicted_ratings_df)
def recommend_items(user_id, user_item_matrix, predicted_ratings_df, num_recommendations=3):
    user_row_number = user_id -1
    sorted_user_ratings = predicted_ratings_df.iloc[user_row_number].sort_values(ascending=False)
    user_data = user_item_matrix.iloc[user_row_number]
    recommendations = sorted_user_ratings[user_data[user_data == 0].index].head(num_recommendations)
    return recommendations

user_id = 1
recommendations = recommend_items(user_id,user_item_matrix, predicted_ratings_df)
print(f"Recommendations for user {user_id}:")
print(recommendations)