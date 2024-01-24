import json
import logging
import os
import argparse

import torch
from transformers import TrainingArguments, Trainer, TrainerCallback, ViTConfig, AutoImageProcessor
from sklearn.metrics import f1_score, accuracy_score, jaccard_score

from data.dataset import load_image_datasets
from data.utils import Vocabulary
from model.modeling_vit import MyViTForImageClassification
from common.logger import setup_logger
from common.compute_metrics import compute_metrics_thre

# TODO
# 5. log only main process /

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='ViT_only')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/code/donghee/inversecooking/data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128) 
    parser.add_argument('--eval_steps', type=int, default=500) 
    parser.add_argument('--logging_steps', type=int, default=50) 
    parser.add_argument('--training_samples', type=int, default=-1, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=1500, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')
    parser.add_argument('--load_from_cache_file', type=bool, default=True, help='load dataset from huggingface cache')
    parser.add_argument('--train_vit', type=bool, default=False, help='train ViT')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='google/vit-base-patch16-224-in21k',
        choices=['google/vit-base-patch16-224-in21k', 'google/vit-large-patch32-224-in21k', 'google/vit-huge-patch14-224-in21k'],
        help="Specifies the model to use. Choose from 'google/vit-base-patch16-224-in21k' (default), 'google/vit-large-patch32-224-in21k', 'google/vit-huge-patch14-224-in21k'. "
    )
    
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)

    return args



def compute_metrics(pred):
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

class PrinterCallback(TrainerCallback): 
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:  # Log at intervals defined by logging_steps
            # Access model and dataloader
            model = kwargs['model']
            dataloader = kwargs['dataloader']

            # Assuming you have a function to get predictions and labels
            predictions, labels = self.get_predictions_and_labels(model, dataloader)

            # Compute metrics
            f1 = self.compute_f1(predictions, labels)
            iou = self.compute_iou(predictions, labels)

            # Log metrics
            print(f"Step {state.global_step}: F1 Score: {f1}, IoU Score: {iou}")


def train(args):
    config = ViTConfig()
    model = MyViTForImageClassification(config, model_name=args.model_name, vit_freeze=args.vit_freeze)
    image_processor = AutoImageProcessor.from_pretrained(args.model_name)

    datasets = load_image_datasets(
        image_processor = image_processor,
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        pre_map=args.pre_map,
        load_from_cache_file = args.load_from_cache_file
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
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="steps",
        save_steps = args.eval_steps,
        save_total_limit=4,
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
        save_safetensors=False,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        compute_metrics=compute_metrics_thre ##
        # data_collator = CustomDataCollator(tokenizer=tokenizer, model=model)
        # callbacks=[PrinterCallback]
    )

    # Train the model
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    eval_result = trainer.evaluate()
    print("EVAL")
    print(eval_result)


    # Save
    # model.save_pretrained_final(os.path.join(args.output_dir, 'best'))

    # print("* Test start *")
    # test_results = trainer.evaluate(datasets['test'])
    # print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)

    ####
    args.training_samples = 128
    args.epochs = 46.5
    args.batch_size = 128
    args.vit_freeze = True
    # args.eval_steps = 10
    # args.eval_samples = -1
    # args.test_samples = 10000
    args.model_name = 'google/vit-huge-patch14-224-in21k'
    args.resume_from_checkpoint = '/nfs_share2/code/donghee/instructBlip/outputs/ViT_only/checkpoint-23000'
    ####

    pretty_print(args)

    train(args)
