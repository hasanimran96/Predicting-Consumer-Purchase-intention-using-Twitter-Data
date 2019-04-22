import math
import pandas as pd
import numpy as np


class DocumentVector:
    def tf_idf(self, df_cleaned_text, unique):
        # unique = list(set(corpus.split()))
        # # print(unique)
        # tfIdf_df = pd.DataFrame(columns=unique)
        tf_df = self.DocVector(df_cleaned_text, unique)
        # idf_df = pd.DataFrame()
        total_docs = len(tf_df.index)
        for column in unique:
            num_doc_word = 0
            for no_doc in range(total_docs):
                if tf_df.at[no_doc, column] > 0:
                    num_doc_word += 1
            # print(num_doc_word)
            idf = math.log(total_docs / (num_doc_word + 1))
            # idf_df.at[0,column] = idf
            tf_df[column] = tf_df[column].multiply(idf)
            idf = 0
        # print(tf_df['buy'])
        return tf_df

    def DocVector(self, final_df, uniqueWords):
        data = np.zeros([final_df["class"].count(), len(uniqueWords)])
        docVector1 = pd.DataFrame(data, columns=uniqueWords)
        docVector = docVector1.assign(PurchaseIntention=list(final_df["class"]))
        # docVector['Purchase Intention'] = final_df['class']
        # print(docVector['PurchaseIntention'])
        doc_count = 0
        for doc in final_df["text"]:
            words = doc.split()
            for word in words:
                temp = word.lower()
                if temp in docVector.columns:
                    docVector.at[doc_count, temp] += 1
            doc_count += 1

        return docVector

    def binary_docvector(self, final_df, uniqueWords):
        data = np.zeros([final_df["class"].count(), len(uniqueWords)])
        docVector1 = pd.DataFrame(data, columns=uniqueWords)
        docVector = docVector1.assign(PurchaseIntention=list(final_df["class"]))
        # docVector['Purchase Intention'] = final_df['class']
        # print(docVector['PurchaseIntention'])
        doc_count = 0
        for doc in final_df["text"]:
            words = doc.split()
            for word in words:
                temp = word.lower()
                if temp in docVector.columns:
                    if docVector.iloc[doc_count][temp] < 1:
                        docVector.at[doc_count, temp] += 1
            doc_count += 1
        # print(docVector['good'])
        return docVector

