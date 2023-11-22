from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration, TrainingArguments, Trainer
# from datasets import Dataset
import torch
# from PIL import Image
import os
from data.data_loader import load_datasets, collator, Recipe1M_Collator
from data.utils import Vocabulary
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration

# TODO
# 1. torch.distriubted run
# 2. instructBlipforConditionalGeneration override 해서 f1 /iou score 내도록 하자 또는 trainer 코드 override 하든가.
# 3. BERT - CLS token 알아보기
# 4. ** optimize Dataset - preprocess.. inefficient to process on the fly

def train():
    ####
    project_name = 'first'
    model_name = 'Salesforce/instructblip-vicuna-7b' # too big
    # model_name = 'Salesforce/instructblip-flan-t5-xl'
    dataset_path = '/nfs_share2/code/donghee/inversecooking/data'
    epochs = 20
    batch_size = 5
    training_samples = 1000 # set to -1 for training on entire dataset
    pre_map = True
    ####

    output_dir= os.path.join("./outputs", project_name)
    logging_dir = os.path.join('./logs', project_name)
    print("* Project name: ", project_name)
    print("* output directory: ", output_dir)
    print("* Log directory: ", logging_dir)

    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(model_name)
    processor = InstructBlipProcessor.from_pretrained(model_name)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    # encoder_decoder = True if 'flan' in model_name else False
    decoder_only = True if model.config.use_decoder_only_language_model else False


    datasets = load_datasets(
        batch_size, 
        processor, 
        data_dir=dataset_path, 
        training_samples=training_samples, 
        pre_map=pre_map,
        decoder_only=decoder_only
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        evaluation_strategy="epoch",
        logging_dir=logging_dir,
        logging_strategy = 'steps',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
    )

    # TODO: compute_metrics
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        # compute_metrics=compute_metrics,
        # data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save
    # processor.save_pretrained(output_dir)
    model.save_pretrained(output_dir)

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)


if __name__ == '__main__':
    train()