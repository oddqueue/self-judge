import warnings
from typing import Dict, List, Optional
from dataclasses import dataclass

import torch

from transformers import PreTrainedTokenizerBase

@dataclass
class JudgeAugmentedSFTCollator:
    def __init__(self,
                 tokenizer: PreTrainedTokenizerBase,
                 max_length: Optional[int] = None,
                 ignore_index: Optional[int] = -100,
                ):

        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        
        # checks max length configurations
        if max_length is None:
            self.max_length = self.tokenizer.model_max_length
            warnings.warn(
                f"You didn't pass a `max_length` argument to the JudgeAugmentedSFTCollator, this will default set to {self.max_length}."
            )
        else:
            self.max_length = max_length

    def __call__(self, examples: Dict[str, List[List[int]]]):
        
        return self.collate_batch(examples)
        
    def collate_batch(self, examples: List[Dict[str, List[int]]]):
        
        input_ids = []
        labels = []

        for example in examples:
            input_ids.append(example['input_ids'])
            labels.append(example['labels'])
        
        padded_inputs = self.tokenizer.pad({'input_ids':input_ids}, return_tensors='pt')
        padded_inputs.input_ids = padded_inputs.input_ids[:, -self.max_length:]
        padded_inputs.attention_mask = padded_inputs.attention_mask[:, -self.max_length:]
        padded_labels = torch.full_like(padded_inputs.input_ids, self.ignore_index)
        
        for sample_index, sample_labels in enumerate(labels):
            sample_labels = sample_labels[-self.max_length:]
            padded_labels[sample_index, -len(sample_labels):] = torch.tensor(sample_labels)

        batch = {
            'input_ids': padded_inputs.input_ids,
            'attention_mask': padded_inputs.attention_mask,
            'labels': padded_labels,
        }
        
        return batch
    
@dataclass
class SelfJudgeCollator:
    def __call__(self, examples: List[Dict[str, str]]):
        
        return self.collate_batch(examples)
            
    def collate_batch(self, examples: List[Dict[str, str]]):
        
        queries = []
        contexts = []
        chosen = []
        rejected = []

        for example in examples:
            queries.append(example['query'])
            contexts.append(example['context'])
            chosen.append(example['chosen'])
            rejected.append(example['rejected'])

        batch = {
            'query': queries,
            'context': contexts,
            'chosen': chosen,
            'rejected': rejected,
        }
        
        return batch