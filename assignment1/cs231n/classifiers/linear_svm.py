from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    
    num_train = X.shape[0]
    num_class = W.shape[1]
    dW = np.zeros(W.shape)
    loss = 0
    
    for i in range(num_train):
        scores = np.dot(X[i], W)
        
        for j in range(num_class):
            if y[i]==j : continue
            margin = scores[j] + 1 - scores[y[i]]
            if margin > 0 : 
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i] 
            
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2*reg*W
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    
    num_train = X.shape[0]
    num_class = W.shape[1]
    
    scores = np.dot(X, W)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1)
    scores = np.maximum(0, scores + 1 - correct_class_score)
    scores[np.arange(num_train), y] = 0
    loss = np.sum(scores) / num_train
    loss += reg*np.sum(W * W)
    
    dW = np.zeros(W.shape)
    scores[scores > 0] = 1
    valid_score = np.sum(scores, axis=1)
    scores[np.arange(num_train), y] -= valid_score
    dW = np.dot(X.T, scores) / num_train
    dW += 2*reg*W
    
    return loss, dW
