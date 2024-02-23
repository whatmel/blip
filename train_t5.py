import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
import json
# import logging
import os
import argparse

from transformers import InstructBlipProcessor, TrainingArguments, EarlyStoppingCallback
from transformers.utils import logging 

from data.dataset import load_datasets
from datasets import load_dataset, DatasetDict
from data.utils import Vocabulary, to_one_hot, remove_unused_columns
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration

from common.logger import setup_logger
from common.compute_metrics import Recipe1mEvalMetrics
from common.trainer import CustomTrainer

# TODO
# 5. log only main process (logging, logger)

logger = logging.get_logger(__name__)

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logger.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='T5_f1')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/shared/from_donghee/recipe1m_data', help='path containing Recipe1M dataset')
    parser.add_argument('--dataset_name', type=str, default='recipe1m', choices=['recipe1m', 'mnist', 'cifar10', 'cifar100'], help='Hugging face built-in datasets or Recipe1M')
    parser.add_argument('--dataset_cache_path', type=str, default='/home/donghee/huggingface_data_cache', help='local dataset cache directory')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--eval_steps', type=int, default=500, help='number of update steps between two evaluations')
    parser.add_argument('--logging_steps', type=int, default=100, help='number of steps between two logs')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--fine_label', type=bool, default=False, help='if True, use fine labels for classification')
    parser.add_argument('--eval_split_ratio', type=float, default=0.1, help='split ratio for validation set')
    parser.add_argument('--generate_mode', type=bool, default=True, help='True for generation task, False for classification task')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )

    args = parser.parse_args()
    
    # args.output_dir= os.path.join("./outputs", args.project_name)
    # args.logging_dir = os.path.join('./logs', args.project_name)

    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):
    """
    Training script for original InstructBlip with T5-xl
    """

    processor = InstructBlipProcessor.from_pretrained(args.model_name)

    possible_cache_dir = os.path.join(args.dataset_cache_path, args.dataset_name)

    if args.dataset_name == 'recipe1m':
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            datasets = remove_unused_columns(datasets, args.generate_mode)
        else:
            logger.info("* Recipe1M mapping start")
            datasets = load_datasets( 
                processor=processor, 
                data_dir=args.dataset_path, 
                training_samples=args.training_samples,
                eval_samples=args.eval_samples, 
                pre_map=args.pre_map,
                decoder_only=args.decoder_only
            )
            datasets = DatasetDict(datasets)
            logger.info("* Recipe1M saving start")
            os.makedirs(possible_cache_dir)
            datasets.save_to_disk(possible_cache_dir)
            logger.info(f"* Save dataset to {possible_cache_dir}") 
        
        num_labels = len(datasets['train'][0]['labels']) ## TODO -2 # 0: <end>, 1487: <pad>
            
    else:
        if os.path.exists(possible_cache_dir):
            logger.info(f"Load {args.dataset_name} from cache")
            datasets = DatasetDict.load_from_disk(possible_cache_dir)
            # TODO make class_names compatible to other datasets (coarse label, fine label..)
            class_names = datasets['train'].features['coarse_label'].names if not args.fine_label else datasets['train'].features['fine_label'].names
            num_labels = len(class_names)
        else:
            datasets = load_dataset(args.dataset_name)
            # TODO make class_names compatible to other datasets (coarse label, fine label..)
            class_names = datasets['train'].features['coarse_label'].names if not args.fine_label else datasets['train'].features['fine_label'].names
            num_labels = len(class_names)
            class_names = ", ".join(class_names).replace('_', ' ')

            def preprocess_data(examples):
                text_input = [f'Identify the main object in the image from the following categories: {class_names}']*len(examples['img'])
                inputs = processor(
                    images = examples['img'],
                    text = text_input,
                    return_tensors='pt',
                ) # input_ids, attention_mask, qformer_iput_ids, qformer_attention_mask, pixel_values

                inputs['labels'] = to_one_hot(examples['coarse_label'] if not args.fine_label else examples['fine_label'], num_classes = num_labels, remove_pad=False) # one-hot labels
                
                return inputs
            
            if len(datasets) == 2: # no eval split
                eval_split_ratio = args.eval_split_ratio # 0.1
                train_test_split = datasets["train"].train_test_split(test_size=eval_split_ratio)
                datasets = DatasetDict({
                    'train': train_test_split['train'],
                    'val': train_test_split['test'],  # new validation set
                    'test': datasets['test']
                })
            
            assert len(datasets) == 3
            datasets = datasets.map(preprocess_data, batched=True)
            
            os.makedirs(possible_cache_dir)
            datasets.save_to_disk(possible_cache_dir)
            logger.info(f"* Save dataset to {possible_cache_dir}") 

    processor.save_pretrained(os.path.join(args.output_dir, 'best'))

    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps, # 500
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=args.logging_steps,
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='f1_micro', # TODO f1_macro? mAP? AP?
        greater_is_better=True,
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        # include_inputs_for_metrics=True,
        remove_unused_columns= False ## TODO
    )

    eval_metrics = Recipe1mEvalMetrics(processor.tokenizer)
    
    trainer = CustomTrainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        tokenizer=processor.tokenizer,
        compute_metrics=eval_metrics.compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Train the model
    trainer.train()

    # eval_result = trainer.evaluate(datasets['val'])
    # print("EVAL")
    # print(eval_result)

    # Save
    model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)

if __name__ == '__main__':
    args = parse_args()

    ###
    # args.batch_size = 20
    # args.training_samples = 100
    # args.eval_samples = 100
    # args.eval_steps = 5
    args.project_name = 'original_instructBlip_t5_recipe1m'
    # args.logging_steps = 50
    ###

    setup_logger(args)
    pretty_print(args)

    train(args)
