import pandas as pd
import numpy as np

path = "data/Annotated4.csv"


def extract(path):
    fd = open(path, encoding="utf-8", errors='replace')
    df = pd.read_csv(fd)
    defined = df['class'] != ("undefined")
    # #output dataframe without undeined
    df2 = df[defined]
    defined1 = df2['class'] != "Undefined"
    df4 = df2[defined1]
    # replace no PI with no
    df3 = df4.replace("No PI", "no")
    # replace PI with yes
    final = df3.replace("PI", "yes")

    replace_yes = final.replace("Yes", "yes")
    final_df = replace_yes.replace("No", "no")
    return final_df, df


def output_to_analysis(path):
    # get the data
    final_data_frame, data_frame_undefined = extract(path)
    # show count of yes/no
    class_count = final_data_frame['class'].value_counts()
    # list of top 25 words in the data
    frequent_words = pd.Series(
        ' '.join(final_data_frame['text']).lower().split()).value_counts()[:25]
    # getting the neg/pos tweets
    negative_tweets = final_data_frame[final_data_frame['class'].isin(['no'])]
    positive_tweets = final_data_frame[final_data_frame['class'].isin([
        'yes'])]
    # converting the neg/pos tweets to str for word cloud
    negative_tweets_str = negative_tweets.text.str.cat()
    positive_tweets_str = positive_tweets.text.str.cat()
    return class_count, frequent_words, negative_tweets_str, positive_tweets_str


class_count, frequent_words, negative_tweets_str, positive_tweets_str = output_to_analysis(
    path)

print(class_count)
print(frequent_words)
print(negative_tweets_str)
print(positive_tweets_str)
