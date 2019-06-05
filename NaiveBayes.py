import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata
from sklearn.metrics import classification_report, precision_score, recall_score


class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1] #CSR matrix has columns as all words of vocab
        self.count_positive = np.zeros([1, self.vocab_len])
        self.count_negative = np.zeros([1, self.vocab_len])
        self.num_positive_reviews = 0
        self.num_negative_reviews = 0
        self.total_positive_words = 0
        self.total_negative_words = 0
        self.P_positive = 0.0
        self.P_negative = 0.0
        self.deno_pos = 0.0
        self.deno_neg =0.0
        self.Train(data.X,data.Y)

    # Train model - X are instances, Y are labels (+1 or -1)
    # X and Y are sparse matrices
    def Train(self, X, Y):
        #TODO: Estimate Naive Bayes model parameters
        positive_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        #num of positive reviews/ pFiles
        self.num_positive_reviews = len(positive_indices)
        #num of negative reviews/ nFiles
        self.num_negative_reviews = len(negative_indices)
         #array of positive counts for each word
        self.count_positive = csr_matrix.sum(X[np.ix_(positive_indices)], axis = 0) + self.ALPHA
        #array of positive counts for each word
        self.count_negative = csr_matrix.sum(X[np.ix_(negative_indices)], axis = 0) + self.ALPHA

        #total count for all positive words
        self.total_positive_words = np.sum(self.count_positive)
        #total count for all negative words
        self.total_negative_words = np.sum(self.count_negative)

        #Deno of P(c) Num of positive words + smoothing factor for all words
        self.deno_pos = self.total_positive_words + self.ALPHA*X.shape[1]
        #Deno of P(c) Num of negative words + smoothing factor for all words
        self.deno_neg = self.total_negative_words + self.ALPHA*X.shape[1]

        # self.count_positive = 1
        # self.count_negative = 1
        self.pos_recall = []
        self.pos_precision = []
        self.neg_recall = []
        self.neg_precision = []

        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X, limit = 0):
        #TODO: Implement Naive Bayes Classification

        #P(C=+) = Num of positive reviews/(Num of documents)
        self.P_positive = log(self.num_positive_reviews) - log(self.num_positive_reviews + self.num_negative_reviews)
        #P(C=+) = Num of negative reviews/(Num of documents)
        self.P_negative = log(self.num_negative_reviews) - log(self.num_positive_reviews + self.num_negative_reviews)
        pred_labels = []
        #Since we are calculating Probabilities in log space hence limit should be in log space as well
        if(limit != 0):
            limit = log(limit)

        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            total_pos_condProb = self.P_positive
            total_neg_condProb = self.P_negative
            for j in range(len(z[0])):
                # Look at each feature
                #pass
                row_index = i
                col_index = z[1][j]
                word_occured = X[row_index,col_index]
                #+ve Prob of P(X|Y=c)
                pos_condProb = log(self.count_positive[0,col_index]) # - log(self.deno_pos)
                #+ve P(X|Y=c)*P(c)
                total_pos_condProb += word_occured*pos_condProb
                #-ve Prob of P(X|Y=c)
                neg_condProb = log(self.count_negative[0,col_index]) #- log(self.deno_neg)
                #+ve P(X|Y=c)*P(c)
                total_neg_condProb += word_occured*neg_condProb
            if(limit != 0):
                if total_pos_condProb > limit:            # Predict positive if greater than limit
                    pred_labels.append(1.0)
                else:               # Predict negative
                    pred_labels.append(-1.0)
            else:
                if total_pos_condProb > total_neg_condProb:            # Predict positive if greater than negative
                    pred_labels.append(1.0)
                else:               # Predict negative
                    pred_labels.append(-1.0)

        #Computing tp,tn,fp,fn required for printing Graphs
        self.tp = 0
        self.tn = 1
        self.fp = 1
        self.fn = 1
        Y= self.data.Y
        for i in range(len(pred_labels)):
            if pred_labels[i] == 1.0:
                if Y[i] == 1.0:
                    self.tp += 1
                else:
                    self.fp += 1
            else:
                if Y[i] == 1.0:
                    self.fn += 1
                else:
                    self.tn += 1

        self.pos_recall.append((self.tp) / (self.tp + self.fn))
        self.neg_recall.append((self.tn) / (self.tn + self.fp))
        self.pos_precision.append((self.tp) / (self.tp + self.fp))
        self.neg_precision.append((self.tn) / (self.tn + self.fn))
        return pred_labels

    def LogSum(self, logx, logy):
        # TO DO: Return log(x+y), avoiding numerical underflow/overflow.
        m = max(logx, logy)
        return m + log(exp(logx - m) + exp(logy - m))

    # Predict the probability of each indexed review in sparse matrix text
    # of being positive
    # Prints results
    def PredictProb(self, test, indexes):

        for i in indexes:
            # TO DO: Predict the probability of the i_th review in test being positive review
            # TO DO: Use the LogSum function to avoid underflow/overflow
            z = test.X[i].nonzero()
            total_pos_condProb = self.P_positive
            total_neg_condProb = self.P_negative
            predicted_label = 0
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                word_occured = test.X[row_index,col_index]
                #+ve Prob of P(X|Y=c)
                pos_condProb = log(self.count_positive[0,col_index]) - log(self.deno_pos)
                #+ve P(X|Y=c)*P(c)
                total_pos_condProb += word_occured*pos_condProb
                #-ve Prob of P(X|Y=c)
                neg_condProb = log(self.count_negative[0,col_index]) - log(self.deno_neg)
                #+ve P(X|Y=c)*P(c)
                total_neg_condProb +=word_occured*neg_condProb

            if total_pos_condProb > total_neg_condProb:
                predicted_label = 1.0
            else:
                predicted_label = -1.0

            predicted_prob_positive = exp(total_pos_condProb - self.LogSum(total_pos_condProb,total_neg_condProb))
            predicted_prob_negative = exp(total_neg_condProb - self.LogSum(total_pos_condProb,total_neg_condProb))

            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative)

    # Evaluate performance on test data
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()

    def EvalPrecision(self, test):
        predicted_Y =np.array(self.PredictLabel(test.X))
        testset_Y = np.array(test.Y)
        print("Precision Score:",precision_score(testset_Y,predicted_Y))
        print("Recall_Score:", recall_score(testset_Y,predicted_Y))

    def EvalRecall(self, test):
        predicted_Y =np.array(self.PredictLabel(test.X))
        testset_Y = np.array(test.Y)
        print("Recall_Score:", recall_score(testset_Y,predicted_Y))

    def topWords(self,X):
        neg_weight = 0
        pos_weight = 0
        weight_dict_neg = {}
        weight_dict_pos = {}
        for i in range(X.shape[0]):
            z = X[i].nonzero()
            for j in range(len(z[0])):
                wordId = z[1][j]
                freq_pos = 1
                if wordId  in self.count_positive:
                    freq_pos = self.count_positive[0,wordId]

                freq_neg = 1
                if wordId in self.count_negative:
                    freq_neg = self.count_negative[0,wordId]

                pos_weight = exp(log(freq_pos) - (log(self.total_positive_words) + self.P_positive))  # log(self.P_positive))
                neg_weight = exp(log(freq_neg) - (log(self.total_negative_words) + self.P_negative))  # log(self.P_negative))

                weight_dict_pos[wordId] = pos_weight / (pos_weight + neg_weight)
                weight_dict_neg[wordId] = neg_weight / (pos_weight + neg_weight)

        weight_dict_pos = sorted(weight_dict_pos.items(), key=lambda x:x[1], reverse=True)
        weight_dict_neg = sorted(weight_dict_neg.items(), key=lambda x:x[1], reverse=True)
        count = 0
        print("Positive Words:")
        for key, value in weight_dict_pos:
            print("(",self.data.vocab.GetWord(key), ": %.4f)" %value, end=",")
            count += 1
            if count == 20:
                break
        print("\n\n")
        print("Negative Words:")
        count = 0
        for key, value in weight_dict_neg:
            print("(",self.data.vocab.GetWord(key), ": %.4f)" %value, end=",")
            count += 1
            if count == 20:
                break


    def plotLimitGraph(self, test):
        x_axis = []
        accuracy = []
        for i in range(9):
            Y_pred = self.PredictLabel(test.X,(i+1)/10)
            ev = Eval(Y_pred, test.Y)
            accuracy.append(ev.Accuracy())
            x_axis.append((i+1)/10)

            # Y_pred = self.PredictLabel(test.X, (i+1)/10)
            # ev = Eval(Y_pred, test.Y)
            # accuracy.append(ev.Accuracy())
            # #Y_pred1 = np.array(Y_pred)
            # #recall_pos.append(recall_score(test.Y,Y_pred1))
            # #precision_pos.append( precision_score(test.Y,Y_pred1))
            # #Y_pred_neg = np.array([1 if i == -1 else -1 for i in Y_pred])
            # #Y_test_neg = np.array([1 if i == -1 else -1 for i in test.Y])
            # #recall_neg.append(recall_score(Y_test_neg,Y_pred_neg))
            # #precision_neg.append(precision_score(Y_test_neg,Y_pred_neg))
            # x_axis.append((i+1)/10)
            # # print(i,recall_pos,precision_pos)

        plt.title('Recall Positive Graph.')
        plt.plot(x_axis, self.pos_recall, label = "Recall Positive")
        plt.plot(x_axis, self.neg_recall, label = "Recall Negative")
        plt.xlabel('Threshold')
        plt.ylabel('Recall')
        plt.title('Recall Plot')
        plt.legend()
        plt.show()

        plt.plot(x_axis, self.pos_precision, label = "Precision Positive")
        plt.plot(x_axis, self.neg_precision, label = "Precision Negative")
        plt.xlabel('Threshold')
        plt.ylabel('Precision')
        plt.title('Precision Plot')
        plt.legend()
        plt.show()


        # plt.plot(precision_pos,recall_pos, label = "Recall vs Precision Positive")
        # plt.plot(precision_neg,recall_neg, label = "Recall vs Precision Negative")
        # plt.xlabel('Precision')
        # plt.ylabel('Recall')
        # plt.title('Precision vs Recall Plot')
        # pl    t.legend()
        # plt.show()

if __name__ == "__main__":

    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    #print("Test Accuracy: ", nb.Eval(testdata))

    #First 10 reviews probability estimates
    print("First 10 reviews probability estimates in test data for ALPHA = ",float(sys.argv[2]))
    range10 = range(10)
    nb.PredictProb(testdata, range10)

    #Computing Recall and Precision score

    nb.EvalPrecision(testdata)
    nb.EvalRecall(testdata)

    #Top Words
    nb.topWords(traindata.X)

    #Plotting graphs
    #Uncomment the below to get graphs
    #nb.plotLimitGraph(testdata)
