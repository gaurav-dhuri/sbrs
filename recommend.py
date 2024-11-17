import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from constants import *
import pickle
import os
import warnings
warnings.simplefilter("ignore")

def get_user_recommendations(username):
    path = 'D:/AI-ML/EPG - AIML/Capstone/SBRS/'
    directory = os.path.join(path, 'pickle/')

    # load user final ratings
    with open(os.path.join(directory,USER_FINAL_RATING), 'rb') as f:
        user_final_rating = pd.read_pickle(f)

    if username in user_final_rating.index:
        # load the preprocessed
        with open(os.path.join(directory,PREPROCESSED_DATA), 'rb') as f:
            preprocessed_data = pd.read_pickle(f)

        # load the vectorizer
        with open(os.path.join(directory,VECTORIZER), 'rb') as f:
            count_vectorizer = pd.read_pickle(f)

        # load the model
        with open(os.path.join(directory,MODEL_PICKLE_FILENAME), 'rb') as f:
            lr_obj = pd.read_pickle(f)

        recommend_df = preprocessed_data
        # get the top 20  recommedation using the user_final_rating
        top20_reco = list(user_final_rating.loc[username].sort_values(ascending=False)[0:20].index)
        # get the product recommedation using the orig data used for trained model
        common_top20_reco = recommend_df[preprocessed_data['id'].isin(top20_reco)]
        # Apply the TFIDF Vectorizer for the given 20 products to convert data in reqd format for modeling
        X = count_vectorizer.transform(common_top20_reco['reviews_complete_text'].values.astype(str))

        # Recommended model was LR SMOTE (with class imbalance handled)
        # So using the same to predict
        common_top20_reco['sentiment_pred'] = lr_obj.predict(X)

        temp_df = common_top20_reco.groupby(by='name').sum()
        # Create a new dataframe "sent_df" to store the count of positive user sentiments
        sent_df = temp_df[['sentiment_pred']]
        sent_df.columns = ['pos_sent_count']
        # Create a column to measure the total sentiment count
        sent_df['total_sent_count'] = common_top20_reco.groupby(by='name')['sentiment_pred'].count()
        # Calclute the positive sentiment percentage
        sent_df['pos_sent_percent'] = np.round(sent_df['pos_sent_count']/sent_df['total_sent_count']*100, 2)
        # Return top 5 recommended products to the user
        result = sent_df.sort_values(by='pos_sent_percent', ascending=False)[:5]
        product_name_list = list(result.index.values)
        return product_name_list
    else:
        print(f"User name {username} doesn't exist")
        return None


# if __name__ == "__main__":
#    get_user_recommendations("rebecca")
