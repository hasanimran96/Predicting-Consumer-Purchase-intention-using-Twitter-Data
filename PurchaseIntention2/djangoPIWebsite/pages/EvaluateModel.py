class Evaluate:
    def confusion_matrix(self, test_data, predict_df):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for i in range(test_data['class'].count()):
            # print(test_data.iloc[i]['class']," ",predict_df.iloc[i]['PredictedClass'] )
            if test_data.iloc[i]['class'] == "yes" and predict_df.iloc[i]['PredictedClass'] == "yes":
                TP += 1
            elif test_data.iloc[i]['class'] == "yes" and predict_df.iloc[i]['PredictedClass'] == "no":
                FN += 1
            elif test_data.iloc[i]['class'] == "no" and predict_df.iloc[i]['PredictedClass'] == "no":
                TN += 1
            elif test_data.iloc[i]['class'] == "no" and predict_df.iloc[i]['PredictedClass'] == "yes":
                FP += 1
        return TP, FN, TN, FP

    def Accuracy(self, TP, TN, FP, FN):
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        # print("Accuracy =",accuracy)
        return accuracy

    def Precision(self, TP, FP):
        precision = TP / (TP + FP)
        # print("Precision = ",precision)
        return precision

    def Recall(self, TP, FN):
        recall = TP / (TP + FN)
        # print("Recall = ",recall)
        return recall

    def fScore(self, TP, FN, FP):
        F1 = 2 * (self.Recall(TP, FN) * self.Precision(TP, FP)) / (self.Recall(TP, FN) + self.Precision(TP, FP))
        # print("f measure",F1)
        return F1

    def TrueNegative(self, TN, FP):
        TrueNeg = TN/(TN+FP)
        return TrueNeg