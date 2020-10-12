from sklearn.metrics import f1_score
from sklearn.metrics import balanced_accuracy_score

def eval_metrics(actual, pred):
    f1score = f1_score(actual, pred)
    balanced_accuracy = balanced_accuracy_score(actual, pred)
    
    return f1score, balanced_accuracy