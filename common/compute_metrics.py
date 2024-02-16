import logging

import torch
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

def compute_metrics_f1(pred):
    labels = pred.label_ids
    preds = pred.predictions
    # TODO
    if len(preds.shape) == 2: # 2D
        preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy() >= 0.5
    else: # 3D
        preds = torch.sigmoid(torch.tensor(pred.predictions[0])).numpy() >= 0.5

    f1_micro = f1_score(labels, preds, average='micro')
    f1_macro = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    iou_macro = jaccard_score(labels, preds, average='macro') # TODO macro iou?
    iou_micro = jaccard_score(labels, preds, average='micro')

    result = {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'accuracy': acc,
        'iou_macro': iou_macro,
        'iou_micro': iou_micro,
    }

    logging.info(f'* Evaluation result: {result}')

    return result

def plot_f1(precisions, recalls, max_f1, max_f1_thre, max_idx, file_name='f1_thre_bert.png'):
    plt.figure(figsize=(8, 6))
    plt.plot(recalls, precisions, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.scatter([recalls[max_idx]], [precisions[max_idx]], color='red')
    plt.text(recalls[max_idx], precisions[max_idx], f'Max F1: {max_f1:.3f} @ Threshold {max_f1_thre}', fontsize=14)

    plt.legend()
    plt.grid(True)
    plt.savefig(file_name)
    print("F1 plot saved at ", file_name)
    plt.show()


def compute_metrics_thre(pred): # TODO enable file name pass
    thresholds = np.arange(0.0, 1.05, 0.05).tolist()
    labels = pred.label_ids
    preds = pred.predictions

    precisions = []
    recalls = []
    f1s = []
    ious = []
    max_f1_thre = 0
    max_f1 = 0
    max_idx = 0
    max_iou = 0
    for idx, thre in enumerate(thresholds):
        if len(preds.shape) == 2: # 2D
            preds = torch.sigmoid(torch.tensor(pred.predictions)).numpy() >= thre
        else: # 3D
            preds = torch.sigmoid(torch.tensor(pred.predictions[0])).numpy() >= thre

        precision = precision_score(labels, preds, average='micro', zero_division=1)
        recall = recall_score(labels, preds, average='micro')

        f1_micro = f1_score(labels, preds, average='micro')
        iou_micro = jaccard_score(labels, preds, average='micro')

        if f1_micro > max_f1:
            max_f1_thre = thre
            max_f1 = f1_micro
            max_idx = idx
            max_iou = iou_micro

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1_micro)
        ious.append(iou_micro)
    
    print(f"** max f1 (threshold={max_f1_thre}): {max_f1}, max iou: {max_iou}")

    # plot_f1(precisions, recalls, max_f1, max_f1_thre, max_idx, file_name='vit_baseline.png') # file name

    # result = {
    #     'thresholds': thresholds,
    #     'precisions': precisions,
    #     'recalls': recalls,
    #     'f1s': f1s,
    #     'ious': ious,
    # }
    result = {
        'max_threshold': max_f1_thre,
        'max_f1': max_f1,
        'max_iou': max_iou
    }

    return result

def compute_metrics_acc(pred):
    labels = pred.label_ids
    logits = pred.predictions
    num_labels = labels.shape[1]
    
    preds = np.argmax(logits, axis=-1)
    preds_one_hot = np.eye(20)[preds]
    acc = accuracy_score(labels, preds_one_hot)
    
    logging.info(f'* Evaluation Accuray: {acc}')

    return {'accuracy': acc} # TODO