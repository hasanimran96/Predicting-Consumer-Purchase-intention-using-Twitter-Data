from textblob import TextBlob


def text_blob(doc_vector, uniqueWords):
    # i = 0
    for word in uniqueWords:
        analysis = TextBlob(word)
        polarity = analysis.sentiment.polarity
        if polarity != 0:
            # for i in range(doc_vector[word].size):
            doc_vector[word] = doc_vector[word].multiply(other=polarity)
            # doc_vector.at[i, word] += polarity

    return doc_vector
