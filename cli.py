# pandas to read the relevant data in 
import pandas as pd
# imports the function to allow for recommendations
from recommender import hybrid_recommendations

def main():
    print('Welcome to this Book Recommendation System!')
    print('===========================================')
    user_id = input("Enter your id to login: ")

    # load the data needed for the system to work
    ratings_data = pd.read_csv('./ratings_100k_weighted_avg.csv')
    predictions_data = pd.read_csv('./predictions_collab.csv')

    # intialise set
    previous_recommendations = set()


    n=10
    # next call the hybrid recommendation function
    hybrid_recs, prev_r = hybrid_recommendations(user_id,ratings_data,n,predictions_data,previous_recommendations)
    print(f"\nTop {n} recommendations for User {user_id}:\n")
    print(hybrid_recs)
    # set to hold previous recommendations, adds the previous recommendations
    previous_recommendations.update(prev_r)

    # then we give the option to change the number of recommendations, e.g. from a default 10 to 20 for example
    while True:
        print('==================================================================')
        choice = input("Do you want to recommend more books? (y/n) ")

        # if not then we exit
        if choice == "n":
            print("Thank you for using, this Book Recommendation System!")
            break
        
        if choice == "y":
            number = input("How many recommendations would you like? ")
            # calls the recommender system again, with the new number of recommendations
            recommendations, previous_r = hybrid_recommendations(user_id, ratings_data,int(number),predictions_data,previous_recommendations)
            print(f"\nTop {number} recommendations for User {user_id}:\n")
            print(recommendations)
            # udpates the set of previous recommendations, so they aren't duplicated
            previous_recommendations.update(previous_r)
if __name__ == '__main__':
    main()