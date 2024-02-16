import torch
import os

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word, idx=None):
        if idx is None:
            if not word in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1
            return self.idx
        else:
            if not word in self.word2idx:
                self.word2idx[word] = idx
                if idx in self.idx2word.keys():
                    self.idx2word[idx].append(word)
                else:
                    self.idx2word[idx] = [word]

                return idx

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<pad>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def load_dataset(p):
    """
    :param p: path to dataset containing ~~~ files
    :return (train_datset, val_dataset, test_dataset)
    """
    pass

def to_one_hot(labels, num_classes=1488, remove_pad = True):
    """
    remove_pad should be False other than Recipe1M, which contains pad class
    """
    labels = torch.tensor(labels)
    one_hot = torch.zeros(torch.tensor(labels).size(0), num_classes)
    # one_hot.scatter_(1, labels, 1) # faster version?

    if labels.dim() == 2: # multi-label classification
        for i, label in enumerate(labels):
            if len(label)==0:
                continue

            one_hot[i, label] = 1
            if remove_pad:
                one_hot[i, [0, num_classes-1]] = 0 # pad remove

    else: # single-label classification
        for i, label in enumerate(labels):
            one_hot[i, label.long()] = 1

    return one_hot

def get_cache_file_name(cache_dir, dataset_name, split_name, fine_label):
    label_type = 'fine' if fine_label else 'coarse'
    return os.path.join(cache_dir, f"{dataset_name}_{split_name}_{label_type}_preprocessed")