import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from scipy.special import softmax


class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.accuracy = 0.0
        self.best_accuracy = 0.0
        self.confusion_matrix = np.zeros((2, 2))

    def Accuracy(self):
        cm = self.Confusion_Matrix()
        TP = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TN = cm[1, 1]

        return (TP + TN) / (TP + FP + FN + TN)

    def Confusion_Matrix(self):
        #return np.sum(self.confusion_matrix, axis=0)
        return self.confusion_matrix
    
    def Sensitivity(self):
        cm = self.Confusion_Matrix()
        TP = cm[0, 0]
        FN = cm[1, 0]

        return TP / (TP + FN)

    def Specificity(self):
        cm = self.Confusion_Matrix()
        TN = cm[1, 1]
        FP = cm[0, 1]

        return TN / (TN + FP)

    def _generate_confusion_matrix(self, gt, pred):
        return confusion_matrix(gt, pred)

    def add_batch(self, gt, pred):
        assert len(gt) == pred.shape[0]
        
        pred = np.squeeze((pred >= 0.5).astype('int'), 1)
        truth = np.squeeze((gt >= 0.5).astype('int'), 1)

        self.confusion_matrix += self._generate_confusion_matrix(truth, pred)

    def reset(self):
        self.accuracy = 0.0
        self.best_accuracy = 0.0
        self.confusion_matrix = np.zeros((2, 2))


if __name__ == "__main__":
    k = 1
    num_classes = 3
    evaluator = Evaluator(num_classes, topk=k)
    gt = torch.LongTensor([1, 0, 1, 1, 0]).cuda().cpu().numpy()
    pred = torch.FloatTensor([[0.1, 1.0, 2.1],
                              [1.0, 0.1, 0.1],
                              [0.1, 1.0, 0.1],
                              [0.1, 1.0, 0.1],
                              [1.0, 0.1, 0.1]]).cuda()
    softmax = nn.Softmax(dim=1)

    pred = softmax(pred).cpu().numpy()

    #evaluator.add_batch(gt, pred)
    #evaluator.add_batch(gt, pred)
    #print(evaluator.Accuracy())
    #print(np.array(evaluator.confusion_matrix).shape)
    #print(evaluator.confusion_matrix)
    #print(np.average(evaluator.confusion_matrix, axis=0))
