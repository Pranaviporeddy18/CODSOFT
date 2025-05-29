import pandas as pd
import numpy as np

# Sample ratings data (user-item matrix)
data = {
    'user_id': ['U1', 'U1', 'U2', 'U2', 'U3', 'U3', 'U4', 'U5'],
    'item_id': ['I1', 'I2', 'I2', 'I3', 'I1', 'I3', 'I2', 'I3'],
    'rating':  [5, 3, 4, 5, 2, 5, 3, 4]
}

df = pd.DataFrame(data)

# Create user-item rating matrix
rating_matrix = df.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)

# Function to compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    dot = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0

# Compute similarity between users
def user_similarity_matrix(rating_matrix):
    users = rating_matrix.index
    sim_matrix = pd.DataFrame(index=users, columns=users)

    for u1 in users:
        for u2 in users:
            sim_matrix.loc[u1, u2] = cosine_similarity(rating_matrix.loc[u1], rating_matrix.loc[u2])
    return sim_matrix.astype(float)

# Get top-N recommendations for a user
def recommend_items(user_id, rating_matrix, top_n=2):
    sim_matrix = user_similarity_matrix(rating_matrix)
    user_ratings = rating_matrix.loc[user_id]
    similarities = sim_matrix[user_id]

    scores = {}
    for other_user in rating_matrix.index:
        if other_user == user_id:
            continue
        similarity = similarities[other_user]
        for item in rating_matrix.columns:
            if user_ratings[item] == 0 and rating_matrix.loc[other_user, item] > 0:
                scores[item] = scores.get(item, 0) + similarity * rating_matrix.loc[other_user, item]

    ranked_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {top_n} recommendations for user {user_id}:")
    for item, score in ranked_items[:top_n]:
        print(f"Item: {item}, Score: {score:.2f}")

# Example usage
recommend_items('U1', rating_matrix)
