

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

def compute_metrics(predictions, ground_truth):
    # Convert predictions to binary using a threshold (e.g., 0.5)
    binary_predictions = (predictions > 0.5).float()

    # True Positives
    TP = (binary_predictions * ground_truth).sum(dim=1)
    
    # False Positives
    FP = (binary_predictions * (1 - ground_truth)).sum(dim=1)
    
    # False Negatives
    FN = ((1 - binary_predictions) * ground_truth).sum(dim=1)

    # Compute metrics for each sample in the batch
    precision = TP / (TP + FP + 1e-10)
    recall = TP / (TP + FN + 1e-10)
    
    f1_score = 2 * precision * recall / (precision + recall + 1e-10)
    iou = TP / (TP + FP + FN + 1e-10)

    # Compute average metrics for the batch
    avg_f1_score = f1_score.mean()
    avg_iou = iou.mean()

    # return avg_f1_score, avg_iou

    return avg_f1_score, avg_iou

