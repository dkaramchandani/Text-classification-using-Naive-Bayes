import sys
from scipy.sparse import csr_matrix
import numpy as np
from Eval import Eval
from math import log, exp
import time
from imdb import IMDBdata

class NaiveBayes:
    def __init__(self, data, ALPHA=1.0):
        self.ALPHA = ALPHA
        self.data = data # training data
        #TODO: Initalize parameters
        self.vocab_len = data.X.shape[1] #CSR matrix has columns as all words of vocab
        self.count_positive = np.zeros([1, vocab_len])
        self.count_negative = np.zeros([1, vocab_len])
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
        negative_indices = np.argwhere(Y == 1.0).flatten()
        negative_indices = np.argwhere(Y == -1.0).flatten()
        #num of positive reviews/ pFiles
        self.num_positive_reviews = len(positive_indices)
        #num of negative reviews/ nFiles
        self.num_negative_reviews = len(negative_indices)
         #array of positive counts for each word
        self.count_positive = csr_matrix.sum(data.X[np.ix_(positive_indices)], axis = 0) + self.ALPHA
        #array of positive counts for each word
        self.count_negative = csr_matrix.sum(data.X[np.ix_(negative_indices)], axis = 0) + self.ALPHA

        #total count for all positive words
        self.total_positive_words = np.sum(self.count_positive)
        #total count for all negative words
        self.total_negative_words = np.sum(self.count_negative)

        #Deno of P(c) Num of positive words + smoothing factor for all words
        self.deno_pos = self.total_positive_words + (self.ALPHA*self.vocab_len)
        #Deno of P(c) Num of negative words + smoothing factor for all words
        self.deno_neg = self.total_negative_words + (self.ALPHA*self.vocab_len)

        # self.count_positive = 1
        # self.count_negative = 1

        return

    # Predict labels for instances X
    # Return: Sparse matrix Y with predicted labels (+1 or -1)
    def PredictLabel(self, X):
        #TODO: Implement Naive Bayes Classification
        #P(C=+) = Num of postivie reviews/(Num of documents)

        self.P_positive = log(self.num_positive_reviews) - log(self.num_positive_reviews + self.num_negative_reviews)
        self.P_negative = log(self.num_negative_reviews) - log(self.num_positive_reviews + self.num_negative_reviews)
        pred_labels = []

        sh = X.shape[0]
        for i in range(sh):
            z = X[i].nonzero()
            for j in range(len(z[0])):
                # Look at each feature
                pass
                row_index = i
                col_index = z[1][j]
                word_occured = X[row_index,col_index]
                #+ve Prob of P(X|Y=c)
                pos_condProb = log(self.count_positive[col_index]) - log(self.deno_pos)
                #+ve P(X|Y=c)*P(c)
                total_pos_condProb = self.P_positive + word_occured*pos_condProb
                #-ve Prob of P(X|Y=c)
                neg_condProb = log(self.count_negative[col_index]) - log(self.nega_pos)
                #+ve P(X|Y=c)*P(c)
                total_neg_condProb = self.P_negative + word_occured*pos_condProb
            if total_pos_condProb>total_neg_condProb:            # Predict positive
                pred_labels.append(1.0)
            else:               # Predict negative
                pred_labels.append(-1.0)

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
            predicted_label = 0
            for j in range(len(z[0])):
                row_index = i
                col_index = z[1][j]
                word_occured = X[row_index,col_index]
                #+ve Prob of P(X|Y=c)
                pos_condProb = log(self.count_positive[col_index]) - log(self.deno_pos)
                #+ve P(X|Y=c)*P(c)
                total_pos_condProb = self.P_positive + word_occured*pos_condProb
                #-ve Prob of P(X|Y=c)
                neg_condProb = log(self.count_negative[col_index]) - log(self.nega_pos)
                #+ve P(X|Y=c)*P(c)
                total_neg_condProb = self.P_negative + word_occured*pos_condProb

            predicted_prob_positive = exp(total_pos_condProb - self.LogSum(total_pos_condProb,total_neg_condProb))
            predicted_prob_negative = exp(total_pos_condProb - self.LogSum(total_pos_condProb,total_neg_condProb))

            if total_pos_condProb > total_neg_condProb:
                predicted_label = 1.0
            else:
                predicted_label = -1.0

            #print test.Y[i], test.X_reviews[i]
            # TO DO: Comment the line above, and uncomment the line below
            print(test.Y[i], predicted_label, predicted_prob_positive, predicted_prob_negative, test.X_reviews[i])

    # Evaluate performance on test data
    def Eval(self, test):
        Y_pred = self.PredictLabel(test.X)
        ev = Eval(Y_pred, test.Y)
        return ev.Accuracy()
    def EvalPrecision(self, test):
        predicted_Y =np.array(self.PredictLabel(test.X))
        testset_Y = np.array(test.Y)
        print("Precision Score:",precision_score(testset_Y,predicted_Y))
        print(recall_score(Y_test,Y_pred))

    def EvalRecall(self, test):
        predicted_Y =np.array(self.PredictLabel(test.X))
        testset_Y = np.array(test.Y)
        print("Recall_Score:", recall_score(testset_Y,predicted_Y))

if __name__ == "__main__":

    print("Reading Training Data")
    traindata = IMDBdata("%s/train" % sys.argv[1])
    print("Reading Test Data")
    testdata  = IMDBdata("%s/test" % sys.argv[1], vocab=traindata.vocab)
    print("Computing Parameters")
    nb = NaiveBayes(traindata, float(sys.argv[2]))
    print("Evaluating")
    print("Test Accuracy: ", nb.Eval(testdata))
    #First 10 reviews probability estimates
    print("Probability estimates for first 10 reviews in test data for ALPHA = ",float(sys.argv[2]))
    range10 = range(10)
    print (nb.PredictProb(test, range10))
    #Computing Recall and Precision score
    nb.EvalPrecision(testdata)
    nb.EvalRecall(testdata)
    #
