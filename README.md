This contains a book recommendation system which recommends 10 meaningful recommendations to a user, 

The dataset used is a Goodreads dataset and can be downloaded from here; https://mengtingwan.github.io/data/goodreads. I used the Mystery, crime and thriller one, which encapsulated reviews, books and interactions.
I mainly used the books and reviews to help inform my recommendations. 

I used a hybrid approach of collaborative and content filtering. Content-based filtering uses a Vector Space Model (VSM) using the TF-IDF weighting scheme and used cosine similarity to rate the similarity of books to a given user 
Collaborative filtering used SVD++, from the module surprise due to having better performance than SVD. The hybrid approach used a weighted average approach to give an equal chance to both methods. 

To further improve the recommendations, an NLP approach was utilised, namely a word2vec model, which gives a better understanding of semantic relationships.

2 measures, precision@k and novelty were used to evaluate the performance of the 2, which proved RS2 gave better performance than that of RS1. 


cli.py and recommender.py allow you to run the basic recommendor on command line

rs2_cli.py and rs2_recommender.py allow you to run the state-of-the-art recommender on command line. 

These can be run by python cli.py, etc. 

Note that both need additional files generated by running the rs_system.ipynb 

Further work, integrate the interactions into the recommendations to help personalise more
