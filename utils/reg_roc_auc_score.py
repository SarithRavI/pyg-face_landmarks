# generic ROC - AUC score calculation 
# for regression task
# from: https://gist.github.com/smazzanti/f3ee53e8b6d4d616ca07207d74c6b812#file-naive_roc_auc_score-py

def naive_roc_auc_score(y_true, y_pred):
    
  num_same_sign = 0
  num_pairs = 0
  
  for a in range(len(y_true)):
    for b in range(len(y_true)):
      if y_true[a] > y_true[b]:
        num_pairs += 1
        if y_pred[a] > y_pred[b]:
          num_same_sign += 1
        elif y_pred[a] == y_pred[b]:
          num_same_sign += .5
        
  return num_same_sign / num_pairs