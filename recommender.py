import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# content based recommendation system
# for td_idf matrix
def create_tf_idf(data):

    # tokenize the text data e.g. descprtion
    # gets rid of the english stop words, to help give a better decsription
    tf_idf = TfidfVectorizer(stop_words='english')

    # fit data and transform to tdidf matrix
    tf_idf_matrix = tf_idf.fit_transform(data['book_description'])

    return tf_idf_matrix

# cold start avoidance in the case of a new user
def cold_start_avoidance(data, n,previous_recommendations):
    # makes sure these aren't already in the set
    popular_books = data[~data['book_id'].isin(previous_recommendations)]
    # gets rid of any duplicates
    popular_books = popular_books.drop_duplicates(subset='title_without_series')
    # sorts the most popular books via weighted avg, to help with the good books with less reviews
    popular_books = data.sort_values(by='weighted_avg', ascending=False)
    # gets the top n books
    popular_n_books = popular_books.head(n)[['book_id', 'title_without_series']]
    # turns into dataframe for easy of use and consistency
    df = pd.DataFrame(popular_n_books, columns=['book_id', 'title_without_series'])
    return df

# function for content-based recommendations
def get_content_recommendations(user_id, data, n, previous_recommendations):
    # first checks if the user is registered/ in the system:
    if user_id not in data['user_id'].unique():
        return cold_start_avoidance(data, n,previous_recommendations)
    
    # creates the tf_idf matrix dependent on the data given
    tf_idf_matrix = create_tf_idf(data)

    
    # Extract books liked by the user, if the id is valid 
    user_books = data[data['user_id'] == user_id]['title_without_series'].tolist()
        
    # Create a DataFrame to store recommendations and their cosine similarities
    recommendations_df = pd.DataFrame(columns=['Cosine Similarity'])

    # Keep track of recommended books to avoid duplicates
    recommended_books_set = set()

    # Look through each book liked by the user and find similar books based on cosine similarities
    for book in user_books:
        # Aggregate TF-IDF vectors for liked books
        liked_book_idx = data.index[(data['title_without_series'] == book) & (data['user_id'] == user_id)].tolist()[0]
        liked_book_tfidf = tf_idf_matrix[liked_book_idx]

        # Calculate cosine similarity between the liked book and all other books
        cosine_similarities = cosine_similarity(liked_book_tfidf, tf_idf_matrix).flatten()

        # Get indices of books sorted by similarity (excluding liked books)
        sim_indices = cosine_similarities.argsort()[::-1]

        # Filter out books the user has already liked and those already recommended
        sim_indices = [i for i in sim_indices if (data['title_without_series'].iloc[i] not in user_books) and 
                                                (data['title_without_series'].iloc[i] not in recommended_books_set)
                                                and(data['title_without_series'].iloc[i] not in previous_recommendations)]

        # Add top n recommendations to the DataFrame, for each book
        top_recommendations = data['title_without_series'].iloc[sim_indices].tolist()
        cosine_sim_values = cosine_similarities[sim_indices]
        book_id = data['book_id'].iloc[sim_indices].tolist()

        recommendations_df = recommendations_df.append(pd.DataFrame({'book_id':book_id, 'title_without_series': top_recommendations, 'Cosine Similarity': cosine_sim_values}))

        recommended_books_set.update(top_recommendations)

    # Sort recommendations by cosine similarity in descending order
    recommendations_df = recommendations_df.sort_values(by=['Cosine Similarity'], ascending=False)

    # Get unique top n recommendations
    unique_recommendations = recommendations_df.drop_duplicates(subset=['title_without_series']).head(n)

    # drops the Cosine Similarity column as it is no longer needed
    unique_recommendations = unique_recommendations.drop('Cosine Similarity', axis=1)

    return unique_recommendations

# collaborative filtering
def make_recommendations(data, n, predictions, user_id, previous_recommendations):
    #pred = pd.DataFrame(predictions)
    # Filter collab_ratings for the specified user_id
    user_ratings = data[data['user_id'] == user_id]
    
    # Merge predictions with user_ratings based on the book_id
    merged_data = pd.merge(predictions, user_ratings, left_on='iid', right_on='book_id')

    # Sorting the merged DataFrame based on estimated ratings
    merged_data.sort_values(by=['est'], inplace=True, ascending=False)

    unique_books_set = set()
    unique_recommendations = []

    for book_id in merged_data['iid']:
        # doesn't contain any duplicate or previous recs
        if book_id not in unique_books_set and book_id not in previous_recommendations:
            unique_books_set.add(book_id)
            unique_recommendations.append(book_id)

    corresponding_titles = data[data['book_id'].isin(unique_recommendations)][['book_id', 'title_without_series']].drop_duplicates()

    return corresponding_titles


# hybrid recommendation system
def hybrid_recommendations(user_id, data, n, predictions,previous_recommendations):
    # call the 2 functions incase they haven't already been run
    content = get_content_recommendations(user_id, data, n,previous_recommendations)
    collab = make_recommendations(data, n, predictions, user_id,previous_recommendations)
    # join the 2 dataframes
    hybrid = pd.concat([content, collab])

    # gets rid of any books that may have already been predicted
    hybrid = hybrid[~hybrid['book_id'].isin(previous_recommendations)]
     
    # use a weighted system based on number of ratings, to decide on the weighting to use for the hybrid approach

    # inspired by this paper: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9590147 

    v = data['ratings_count']
    R = data['book_average_rating']

    C = data['book_average_rating'].mean()

    m = data['ratings_count']>25
    
    data['weighted_avg'] = ((R*v) + (C*m)) / (v+m)

    # change column type of book_id to be a integer, as it was a float, so hard to look up the weighted average
    hybrid['book_id'] = hybrid['book_id'].astype(int)
    
    # look at how to use the book_id to then find the weighted average and then add this to hybrid table 
    id_weight = data.set_index('book_id')['weighted_avg'].to_dict()

    # Map the weighted_avg values to the 'book_id' column in the hybrid system
    hybrid['weighted_avg'] = hybrid['book_id'].map(id_weight)

    # Sort the DataFrame by 'weighted_avg' in descending order
    hybrid_sorted = hybrid.sort_values(by='weighted_avg', ascending=False)

    # Select the top n rows with unique values
    top_n_unique_values = hybrid_sorted.head(n).drop_duplicates(subset='title_without_series')
    # gets rid of the weighted_avg column to allow for user friendly format
    
    values = top_n_unique_values.drop('weighted_avg',axis=1)

    # updates so that previous recommendations aren't given
    previous_recommendations.update(values['book_id'].tolist())

    print('here')
    
    # check incase we can't get enough recommendations due to content and collab not producing anymore
    if len(values) <n:
        needed = n - len(values)
        # makes sure these aren't already in the set
        possible_recs = data[~data['book_id'].isin(previous_recommendations)]
        # drops duplicates
        possible_recs = possible_recs.drop_duplicates(subset='title_without_series')
        # sorts the rest by the weighted avg, as want the best ones of these
        possible_recs = possible_recs.sort_values(by='weighted_avg', ascending=False)
        # takes the top needed values
        new_values = possible_recs.head(needed)[['book_id', 'title_without_series']]
        #new_values = new_values[~new_values['book_id'].isin(previous_recommendations)]
        # concat with the previous values
        values = pd.concat([values, new_values])

    return values, previous_recommendations
