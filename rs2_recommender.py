import numpy as np
import pandas as pd

# takes the average all previously rated items and produces a resultant vector
def aggregate_vectors(products, w2_model):
    product_vec = []
    for i in products:
        try:
            product_vec.append(w2_model.wv[i])
        except KeyError:
            continue
    if product_vec:
        return np.mean(product_vec, axis=0)
    else:
        return np.zeros(w2_model.vector_size)

def content_rs2(user_id, data, n,w2_model, previous_recommendations):
    
    # extract liked books by user
    user_books = data[data['user_id'] == user_id]['book_id'].tolist()
    # calculate the resultant vector
    user_vector = aggregate_vectors(user_books, w2_model)

    #print(user_books, user_vector)

    # extract most similar products for the input vector, adds the length of previous recommednations as well
    # as these are still most similar, want the next n tho!
    ms = w2_model.wv.most_similar([user_vector], topn=n + len(previous_recommendations))
    # extracts the top n most similar books, excluding itself and the previous recommendations
    book_ids = [int(j[0]) for j in ms if int(j[0]) not in previous_recommendations]
    # creates a df of the most similar book_ids
    book_ids_df = pd.DataFrame({'book_id': book_ids})
    # merges the dataset with 

    new_ms = pd.merge(book_ids_df, data, how='left', on='book_id')
    # gets rid of the duplicated values when merging, can run into problems due to having mutliple instances 
    # of same book_id
    new_ms.drop_duplicates(inplace=True, subset='book_id', keep="last")
    
    # returns the dataframe but only the book_id and title column
    return new_ms[['book_id', 'title_without_series']]


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
def hybrid_recommendations(user_id, data, n, predictions,previous_recommendations, w2_model):
    # call the 2 functions incase they haven't already been run
    content = content_rs2(user_id, data, n, w2_model,previous_recommendations)
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
