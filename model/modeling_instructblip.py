import os
from typing import Callable, Optional, Tuple, Union
import warnings

import torch
from transformers import (
    InstructBlipForConditionalGeneration, 
    InstructBlipConfig, 
    BertForSequenceClassification,
    T5ForConditionalGeneration
)
from transformers import InstructBlipVisionModel, InstructBlipQFormerModel
from transformers.models.instructblip.modeling_instructblip import InstructBlipForConditionalGenerationModelOutput
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from .modeling_t5 import OrderInvariantT5ForConditionalGeneration

class BERTInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    # TODO
    config_class = InstructBlipConfig
    main_input_name = "pixel_values"

    def __init__(self, bert_name, train_llm=False, train_vit=False):
        # 1. vision_model 2. language_model 3. qformer 4. query_tokens 5. language_projection
        config = InstructBlipConfig()

        super().__init__(config)
        # Done initialization - qformer, query_tokens

        self.bert_name = bert_name
        self.num_labels = 1488
        self.qformer_hidden_size = config.qformer_config.hidden_size

        language_model = BertForSequenceClassification.from_pretrained(bert_name, problem_type="multi_label_classification", num_labels=self.num_labels)
        # bert-base-uncased: 110M

        self.language_projection = nn.Linear(self.qformer_hidden_size, language_model.config.hidden_size)
        
        self.post_init() # re-init language projection layer

        # overwrite ViT, BERT weight
        self.vision_model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").vision_model

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model
        self.backbone_freeze(train_llm, train_vit)

    def to_bert(self, bert_name, train_llm=False):
        self.num_labels = 1488

        language_model = BertForSequenceClassification.from_pretrained(bert_name, problem_type="multi_label_classification", num_labels=self.num_labels)
        # bert-base-uncased: 110M

        self.language_projection = nn.Linear(self.qformer_hidden_size, language_model.config.hidden_size)

        if language_model._no_split_modules is not None:
            self._no_split_modules.extend(language_model._no_split_modules)

        if language_model._keep_in_fp32_modules is not None:
            self._keep_in_fp32_modules.extend(language_model._keep_in_fp32_modules)

        self.language_model = language_model

        self.backbone_freeze(train_llm)
    
    def backbone_freeze(self, train_llm=False, train_vit=False):

        if not train_vit:
            for param in self.vision_model.parameters():
                param.requires_grad = False

        if not train_llm:
            for param in self.language_model.parameters():
                param.requires_grad = False
        
        # train classifier
        for param in self.language_model.classifier.parameters():
            param.requires_grad = True

        self.set_ignore_keys(train_llm, train_vit)

    def set_ignore_keys(self, train_llm, train_vit, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        if train_llm:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('language_model')
            ignore_prefix = tuple(ignore_prefix)
        
        if train_vit:
            ignore_prefix = list(ignore_prefix)
            ignore_prefix.remove('vision_model')
            ignore_prefix = tuple(ignore_prefix)
        
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        
        # include classifier to trainable param
        if not train_llm:
            ignore_keys.remove('language_model.classifier.weight')
            ignore_keys.remove('language_model.classifier.bias')

        self._keys_to_ignore_on_save = set(ignore_keys)
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save the model using the traditional PyTorch way (pickle) by safe_serialization to False
        """
        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )
    
    def save_pretrained_final(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save entire model (including ViT, LLM)
        """
        self._keys_to_ignore_on_save = None

        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            labels=labels,
        )
        loss = outputs.loss if return_dict else outputs[0]
        logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )


class FreezeInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, config: InstructBlipConfig):
        super().__init__(config)

        for param in self.vision_model.parameters():
            param.requires_grad = False

        for param in self.language_model.parameters():
            param.requires_grad = False

        self.set_ignore_keys()

    def set_ignore_keys(self, ignore_prefix=('vision_model', 'language_model')):
        """
        Set the _keys_to_ignore_on_save attribute of the model to ignore all keys except those starting with the Q-former prefix.

        Arguments:
            model (PreTrainedModel): The model whose keys are to be filtered.
            qformer_prefix (str): The prefix used for the Q-former's parameters.
        """
        all_keys = self.state_dict().keys()
        ignore_keys = [key for key in all_keys if key.startswith(ignore_prefix)]
        self._keys_to_ignore_on_save = set(ignore_keys)

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save the model using the traditional PyTorch way (pickle) by safe_serialization to False
        """
        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )
    
    def save_pretrained_final(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: bool = True,
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = True,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        **kwargs,
    ):
        """
        Save entire model (including ViT, LLM)
        """
        self._keys_to_ignore_on_save = None
        
        super().save_pretrained(
            save_directory=save_directory, 
            is_main_process=is_main_process,
            state_dict=state_dict,
            save_function=save_function,
            push_to_hub=push_to_hub,
            max_shard_size=max_shard_size,
            safe_serialization=False,
            variant=variant,
            token=token,
            save_peft_format=save_peft_format,
            **kwargs
        )


class OrderInvariantInstructBlipForConditionalGeneration(InstructBlipForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def to_order_invariant_t5(self, model_name):
        self.language_model = OrderInvariantT5ForConditionalGeneration.from_pretrained(model_name)
    
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: torch.FloatTensor,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        positions=None,
    ) -> Union[Tuple, InstructBlipForConditionalGenerationModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size -
            1]`. All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
        >>> processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

        >>> device = "cuda" if torch.cuda.is_available() else "cpu"
        >>> model.to(device)  # doctest: +IGNORE_RESULT

        >>> url = "https://raw.githubusercontent.com/salesforce/LAVIS/main/docs/_static/Confusing-Pictures.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> prompt = "What is unusual about this image?"
        >>> inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        >>> outputs = model.generate(
        ...     **inputs,
        ...     do_sample=False,
        ...     num_beams=5,
        ...     max_length=256,
        ...     min_length=1,
        ...     top_p=0.9,
        ...     repetition_penalty=1.5,
        ...     length_penalty=1.0,
        ...     temperature=1,
        ... )
        >>> generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        >>> print(generated_text)
        The unusual aspect of this image is that a man is ironing clothes on the back of a yellow SUV, which is parked in the middle of a busy city street. This is an unconventional approach to ironing clothes, as it requires the man to balance himself and his ironing equipment on top of the vehicle while navigating through traffic. Additionally, the presence of taxis and other vehicles in the scene further emphasizes the unusual nature of this situation.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # step 1: forward the images through the vision encoder,
        # to get image embeddings of shape (batch_size, seq_len, hidden_size)
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        image_embeds = vision_outputs[0]

        # step 2: forward the query tokens through the QFormer, using the image embeddings for cross-attention
        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        # difference with BLIP-2 here: we also feed the instruction prompt to the Q-Former
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0][:, : query_tokens.size(1), :]

        # step 3: use the language model, conditioned on the query outputs and the prompt
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_model_attention_mask.to(attention_mask.device), attention_mask], dim=1)

        if self.config.use_decoder_only_language_model:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits if return_dict else outputs[0]
            loss = None
            # we compute the loss here since we need to take into account the sequence length of the query embeds
            if labels is not None:
                labels = labels.to(logits.device)
                logits = logits[:, -labels.size(1) :, :]
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous().to(logits.device)

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="mean")

                loss = loss_fct(shift_logits.view(-1, self.config.text_config.vocab_size), shift_labels.view(-1))
        else:
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                labels=labels,
                positions=positions ##
            )
            loss = outputs.loss if return_dict else outputs[0]
            logits = outputs.logits if return_dict else outputs[1]

        if not return_dict:
            output = (logits, vision_outputs, query_outputs, outputs)
            return ((loss,) + output) if loss is not None else output

        return InstructBlipForConditionalGenerationModelOutput(
            loss=loss,
            logits=logits,
            vision_outputs=vision_outputs,
            qformer_outputs=query_outputs,
            language_model_outputs=outputs,
        )

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.FloatTensor,
        qformer_input_ids: Optional[torch.LongTensor] = None,
        qformer_attention_mask: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:
        """
        Overrides `generate` function to be able to use the model as a conditional generator.

        Args:
            pixel_values (`torch.FloatTensor` of shape (batch_size, num_channels, height, width)):
                Input images to be processed.
            qformer_input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt to be fed to the Q-Former module.
            qformer_attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.
            input_ids (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.LongTensor` of shape (batch_size, sequence_length), *optional*):
                Mask to avoid performing attention on padding token indices.

        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        if hasattr(self, "hf_device_map"):
            # preprocess for `accelerate`
            self._preprocess_accelerate()

        batch_size = pixel_values.shape[0]
        image_embeds = self.vision_model(pixel_values, return_dict=True).last_hidden_state

        image_attention_mask = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=image_embeds.device)
        if qformer_attention_mask is None:
            qformer_attention_mask = torch.ones_like(qformer_input_ids)
        qformer_attention_mask = torch.cat([query_attention_mask, qformer_attention_mask], dim=1)
        query_outputs = self.qformer(
            input_ids=qformer_input_ids,
            attention_mask=qformer_attention_mask,
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            return_dict=True,
        )
        query_output = query_outputs.last_hidden_state[:, : query_tokens.size(1), :]

        language_model_inputs = self.language_projection(query_output)
        language_attention_mask = torch.ones(
            language_model_inputs.size()[:-1], dtype=torch.long, device=language_model_inputs.device
        )

        if input_ids is None:
            input_ids = (
                torch.LongTensor([[self.config.text_config.bos_token_id]])
                .repeat(batch_size, 1)
                .to(image_embeds.device)
            )
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        attention_mask = torch.cat([language_attention_mask, attention_mask.to(language_attention_mask.device)], dim=1)

        # concatenate query embeddings with prompt embeddings
        inputs_embeds = self.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([language_model_inputs, inputs_embeds.to(language_model_inputs.device)], dim=1)

        outputs = self.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            **generate_kwargs,
        )

        # the InstructBLIP authors used inconsistent tokenizer/model files during training,
        # with the tokenizer's bos token being set to </s> which has ID=2,
        # whereas the model's text config has bos token id = 0
        if self.config.text_config.architectures[0] == "LLaMAForCausalLM":
            if isinstance(outputs, torch.Tensor):
                outputs[outputs == 0] = 2
            else:
                outputs.sequences[outputs.sequences == 0] = 2

        return outputs
    
