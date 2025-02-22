from enum import Enum

class ScoringMetric(Enum):
    ACCURACY = "accuracy"
    PRECISION = "precision"
    RECALL = "recall"
    F1 = "f1"
    ROC_AUC = "roc_auc"
    NEGATIVE__MEAN_SQUARED_ERROR = 'neg_mean_squared_error'
    
def toString(self):
    return self.value   
 
def scoring_metric_toString(scoring: ScoringMetric):
    print(f"Scoring will use {scoring.value}.")