from transformers import InstructBlipProcessor, AutoTokenizer, CLIPProcessor

class BERTInstructBlipProcessor(InstructBlipProcessor):
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)

    
    def to_bert(self, bert_name):
        # BERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(bert_name) # TODO

class CLIP_QT5InstructBlipProcessor(InstructBlipProcessor):
    
    # TODO
    attributes = ["image_processor", "tokenizer"]
    image_processor_class = "BlipImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, image_processor, tokenizer, qformer_tokenizer):
        super().__init__(image_processor, tokenizer, qformer_tokenizer)

    
    def to_clip(self, clip_model = 'openai/clip-vit-large-patch14-336'):
        self.image_processor = CLIPProcessor.from_pretrained(clip_model)
