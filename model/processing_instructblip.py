from transformers import InstructBlipProcessor, AutoTokenizer

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