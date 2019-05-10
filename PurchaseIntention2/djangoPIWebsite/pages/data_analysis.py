import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords

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
 
def remove_punc(temp_df):
        count = 0
        for text in temp_df['text']:
            out = re.sub(r'[^\w\s]', '', text)
            temp_df.at[count,'text'] = out
            temp_df.at[count, 'class'] = temp_df.iloc[count]['class']
            count += 1
        return temp_df


def output_to_analysis(path):
    # get the data
    final_data_frame, data_frame_undefined = extract(path)
    # show count of yes/no
    # final_data_frame = remove_punc(final_data_frame)
    final_data_frame_temp = final_data_frame.iloc[0:100]
    class_count = final_data_frame['class'].value_counts()
    stop = stopwords.words("english")
    stop.append("I")
    stop.append("X")
    stop.append("i")
    stop.append("x")
    stop.append(".")
    stop.append("@")
    final_data_frame["text"] = final_data_frame["text"].apply(
        lambda x: " ".join(x for x in x.split() if x not in stop)
    )
    # list of top 25 words in the data
    frequent_words = pd.Series(
        ' '.join(final_data_frame['text']).lower().split()).value_counts()[:25]
    # getting the neg/pos tweets
    negative_tweets = final_data_frame_temp[final_data_frame_temp['class'].isin(['no'])]
    positive_tweets = final_data_frame_temp[final_data_frame_temp['class'].isin([
        'yes'])]
    # converting the neg/pos tweets to str for word cloud
    negative_tweets_str = negative_tweets.text.str.cat().replace('\n', ' ')
    positive_tweets_str = positive_tweets.text.str.cat().replace('\n', ' ')
    print(negative_tweets_str)
    return class_count, frequent_words, negative_tweets_str, positive_tweets_str

# output_to_analysis("uploadeddata\Annotated4.csv")
