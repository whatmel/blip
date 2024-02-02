import json
import logging
import os
import argparse
import pickle
from typing import Dict
import random

import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import InstructBlipProcessor, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import load_datasets, CustomDataCollator, collator, Recipe1M_Collator, load_datasets_for_distributed
from data.utils import Vocabulary, to_one_hot
from model.modeling_instructblip import QT5InstructBlipForConditionalGeneration
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre

from transformers.trainer_utils import EvalPrediction

# TODO
# 5. log only main process

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='T5_learnable_query')
    # /path/to/Recipe1M/dataset
    # /nfs_share2/shared/from_donghee/recipe1m_data
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1000, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--num_query', type=int, default=8, help='number of learnable query passed to decoder')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Call the parent method
        batch = super().__call__(features)

        # Ensure ingredient_ids are included in the batch
        ingredient_ids = [feature['ingredient_ids'] for feature in features]
        batch['ingredient_ids'] = ingredient_ids

        return batch

def compute_metrics(pred): # TODO location
    labels = pred.label_ids
    preds = pred.predictions

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

def train(args):
    model = QT5InstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    # TODO better way to reinit
    model.reinit(num_query=args.num_query)
    processor = InstructBlipProcessor.from_pretrained(args.model_name)
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    datasets = load_datasets( 
        processor=processor, 
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        pre_map=args.pre_map,
        decoder_only=args.decoder_only
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        do_train=False, ## True !!
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        # remove_unused_columns= False ## TODO
    )
    # print("No TRAIN!!")
    # TODO: compute_metrics
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        # data_collator = CustomDataCollator(tokenizer=processor.tokenizer, model=model)
        compute_metrics=compute_metrics_thre, ## compute_metrics
        # data_collator=data_collator
        # metric_class = eval_metrics,
    )

    # Train the model
    trainer.train()

    eval_result = trainer.evaluate(datasets['val'])
    print("EVAL")
    print(eval_result)

    # Save
    # processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ###
    args.batch_size = 16
    args.training_samples = 32
    args.eval_samples = 32
    # args.eval_steps = 200
    # args.logging_steps = 50
    args.epochs=30
    args.num_query = 1
    # args.project_name = 't5_learnable_query1'
    args.project_name = 'temp'
    # args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/T5_learnable_query16/best'
    ###

    pretty_print(args)

    train(args)
