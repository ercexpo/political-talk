import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score

y_true, y_preds = [], []

i = 0
with open('/content/predictions.csv', encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if i != 0:
            y_preds.append(int(row[2]))
            y_true.append(int(row[4]))
        i += 1
        
        
print('Accuracy: ', accuracy_score(y_true, y_preds))
print('Precision: ', precision_score(y_true, y_preds, average='macro'))
print('Recall: ', recall_score(y_true, y_preds, average='macro'))