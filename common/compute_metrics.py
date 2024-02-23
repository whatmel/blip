# import logging
from transformers.utils import logging
import pickle
import random

import torch
from sklearn.metrics import f1_score, accuracy_score, jaccard_score, precision_score, recall_score
import numpy as np
import matplotlib.pyplot as plt

from data.utils import Vocabulary, to_one_hot

logger = logging.get_logger(__name__)

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

    logger.info(f'* Evaluation result: {result}')

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
    
    logger.info(f'* Evaluation Accuray: {acc}')

    return {'accuracy': acc} # TODO

class Recipe1mEvalMetrics():

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ingr2id = pickle.load(open('/nfs_share2/shared/from_donghee/recipe1m_data/recipe1m_vocab_ingrs.pkl', 'rb')).word2idx

    def map_to_classes(self, batch_tokens, max_len=20, show_gen_examples=False):
        ingredient_text = self.tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)
        
        # Process all ingredients in a batch together
        batch_ingr_ids = []
        for ingrs in ingredient_text:
            ingr_text = [ingr.strip().replace(' ', '_') for ingr in ingrs.split(',')]
            ingr_ids = [self.ingr2id.get(ingr, None) for ingr in ingr_text if ingr in self.ingr2id]
            # batch_ingr_ids.append(ingr_ids)

            # Pad the list to ensure consistent length
            if max_len > len(ingr_ids):
                padded_ingr_ids = ingr_ids + [self.ingr2id.get("<pad>", -1)] * (max_len - len(ingr_ids))
            else:
                padded_ingr_ids = ingr_ids
            
            if show_gen_examples and random.random() < 0.02:
                logger.info(f"* Generation example: {ingrs}")
            
            batch_ingr_ids.append(padded_ingr_ids[:max_len])  # Ensures the list is not longer than max_len

        return batch_ingr_ids

    def compute_metrics(self, generation_ids, ingredient_ids_one_hot, show_gen_examples=False, verbose=True): 
        
        target_ingr = ingredient_ids_one_hot

        pred_ingr = self.map_to_classes(generation_ids)    
        pred_ingr = to_one_hot(torch.tensor(pred_ingr))
        
        f1_micro = f1_score(target_ingr, pred_ingr, average='micro')
        f1_macro = f1_score(target_ingr, pred_ingr, average='macro')
        iou_micro = jaccard_score(target_ingr, pred_ingr, average='micro')
        iou_macro = jaccard_score(target_ingr, pred_ingr, average='macro')

        result = {
            'f1_micro': f1_micro,
            'f1_macro': f1_macro,
            'iou_macro': iou_macro,
            'iou_micro': iou_micro,
        }

        if verbose:
            logger.info(f'* Evaluation result: {result}')

        return result