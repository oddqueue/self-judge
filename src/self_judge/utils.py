import os
import re
import warnings
import random
from typing import Dict, List, Optional, Union
from contextlib import contextmanager

import numpy as np

import torch
import torch.nn as nn

from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback


def get_conditional_labels(labels: Union[List[List[int]], torch.Tensor],
                           user_template_ids: List[int],
                           assistant_template_ids: List[int],
                           ignore_index: Optional[int] = -100,
                           last_only: Optional[bool] = False,
                          ):
    
    end_indices = []
    for index in np.where(np.array(labels) == assistant_template_ids[0])[0]:
        if (index+len(assistant_template_ids)) <= len(np.array(labels)):
            if (np.array(labels)[index:index+len(assistant_template_ids)] == np.array(assistant_template_ids)).all():
                end_indices.append(index+len(assistant_template_ids))

    if len(end_indices) == 0:
        warnings.warn(
            f"Could not find the assistant template."
            f" This instance will be ignored in loss calculation."
        )
        for index in range(len(labels)):
            labels[index] = ignore_index
        length = np.nan
            
    elif last_only:
        start_index, end_index = 0, end_indices[-1]
        for index in range(start_index, end_index):
            labels[index] = ignore_index
        length = end_index - start_index
        
    else:
        start_indices = []
        for index in np.where(np.array(labels) == user_template_ids[0])[0]:
            if index+len(user_template_ids) <= len(np.array(labels)):
                if (np.array(labels)[index:index+len(user_template_ids)] == np.array(user_template_ids)).all():
                    start_indices.append(index)

        if len(start_indices) == 0:
            warnings.warn(
                f"Could not find the user template."
                f" This instance will be ignored in loss calculation."
            )
            for index in range(len(labels)):
                labels[index] = ignore_index
            length = np.nan
            
        elif len(start_indices) != len(end_indices):
            warnings.warn(
                f"The number of user/assistant roles does not match."
                f" This instance will be ignored in loss calculation."
            )
            for index in range(len(labels)):
                labels[index] = ignore_index
            length = np.nan
                
        else:
            start_indices[0] = 0
            length = []
            for start_index, end_index in zip(start_indices, end_indices):
                for index in range(start_index, end_index):
                    labels[index] = ignore_index
                length.append(end_index - start_index)
            length = np.mean(length)    
            
    return labels, length


def get_batch_judgments(model: Union[PreTrainedModel, nn.Module],
                        tokenizer: PreTrainedTokenizerBase,
                        judge_templates: Dict[str, Dict[str, str]],
                        token_id_a: int,
                        token_id_b: int,
                        contexts: List[str],
                        responses: List[List[str]],
                        max_length: int,
                        judge_batch_size: int=1,):
    
    for principle, judge_template in judge_templates.items():
        if not bool(re.search(r'\{context\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `context` format. Please ensure this format on your judgment template.")

        if not bool(re.search(r'\{response_a\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_a` format. Please ensure this format on your judgment template.")

        if not bool(re.search(r'\{response_b\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_b` format. Please ensure this format on your judgment template.")

    judge_prompts = []
    judge_types = []
    
    for context_index, (context, (response_a, response_b)) in enumerate(zip(contexts, responses)):
        for principle_index, judge_template in enumerate(judge_templates.values()):
            judge_user_a = judge_template['user'].format(context=context,
                                                         response_a=response_a,
                                                         response_b=response_b,)
            judge_user_b = judge_template['user'].format(context=context,
                                                         response_a=response_b,
                                                         response_b=response_a,)
            
            judge_messages_a = [{"role": "system", "content": judge_template['system']},
                                {"role": "user", "content": judge_user_a},
                                {"role": "assistant", "content": judge_template['assistant']}]
            judge_messages_b = [{"role": "system", "content": judge_template['system']},
                                {"role": "user", "content": judge_user_b},
                                {"role": "assistant", "content": judge_template['assistant']}]
            
            judge_prompt_a = tokenizer.apply_chat_template(judge_messages_a, tokenize=False)
            judge_prompt_b = tokenizer.apply_chat_template(judge_messages_b, tokenize=False)
            
            if judge_prompt_a.endswith(tokenizer.eos_token):
                judge_prompt_a = judge_prompt_a[:-len(tokenizer.eos_token)]
            if judge_prompt_b.endswith(tokenizer.eos_token):
                judge_prompt_b = judge_prompt_b[:-len(tokenizer.eos_token)]
                
            judge_prompts.append(judge_prompt_a)
            judge_prompts.append(judge_prompt_b)
            
            judge_types.append((context_index, principle_index, 0))
            judge_types.append((context_index, principle_index, 1))

    judge_scores = torch.zeros(len(contexts), len(judge_templates), 2, 2)
    
    for judge_batch_index in range(0, len(judge_prompts), judge_batch_size):
        encoded_inputs = tokenizer(judge_prompts[judge_batch_index:judge_batch_index+judge_batch_size],
                                        padding=True,
                                        truncation=True,
                                        max_length=max_length,
                                        return_tensors="pt").to(model.device)
        
        outputs = model.generate(**encoded_inputs,
                                    return_dict_in_generate=True,
                                    output_scores=True,
                                    max_new_tokens=1)
        
        scores = torch.softmax(outputs.scores[0], dim=-1)
        scores_a = scores[:, token_id_a]
        scores_b = scores[:, token_id_b]
        
        for judge_type, score_a, score_b in zip(judge_types[judge_batch_index:judge_batch_index+judge_batch_size], scores_a, scores_b):
            judge_scores[judge_type[0], judge_type[1], judge_type[2], 0] = score_a.item()
            judge_scores[judge_type[0], judge_type[1], judge_type[2], 1] = score_b.item()
            
    judge_scores_a = (judge_scores[:, :, 0, 0] + judge_scores[:, :, 1, 1]) / 2
    judge_scores_b = (judge_scores[:, :, 0, 1] + judge_scores[:, :, 1, 0]) / 2
    
    chosens = []
    rejecteds = []
    
    for (response_a, response_b), judge_score_a, judge_score_b in zip(responses, judge_scores_a, judge_scores_b):    
        win_rates = ((judge_score_a > judge_score_b).sum() / len(judge_templates))
        
        if win_rates > 0.5:
            chosen = response_a
            rejected = response_b
        elif win_rates < 0.5:
            chosen = response_b
            rejected = response_a
        else:
            if judge_score_a.mean() > judge_score_b.mean():
                chosen = response_a
                rejected = response_b
            else:
                chosen = response_b
                rejected = response_a
                
        chosens.append(chosen)
        rejecteds.append(rejected)
    
    return chosens, rejecteds


@contextmanager
def gradient_checkpointing_context(
    model: Union[PreTrainedModel, nn.Module],
    is_gradient_checkpointing: bool,
    gradient_checkpointing_kwargs: Optional[Dict]=None,
    ):
    
    try:
        if is_gradient_checkpointing:
            if hasattr(model, 'module'):
                model.module.gradient_checkpointing_disable()
            else:
                model.gradient_checkpointing_disable()
        yield
    finally:
        if is_gradient_checkpointing:
            if hasattr(model, 'module'):
                model.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
            else:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
                

class PeftSavingCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if args.should_save:
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            kwargs["model"].save_pretrained(checkpoint_path)

            if "pytorch_model.bin" in os.listdir(checkpoint_path):
                os.remove(os.path.join(checkpoint_path, "pytorch_model.bin"))
                

def tournament_rejection_sampling(model: Union[PreTrainedModel, nn.Module],
                                  tokenizer: PreTrainedTokenizerBase,
                                  judge_templates: Dict[str, Dict[str, str]],
                                  token_id_a: int,
                                  token_id_b: int,
                                  contexts: List[str],
                                  responses: List[List[str]],
                                  max_length: int,
                                  judge_batch_size: int=1,):
    
    for principle, judge_template in judge_templates.items():
        if not bool(re.search(r'\{context\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `context` format. Please ensure this format on your judgment template.")

        if not bool(re.search(r'\{response_a\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_a` format. Please ensure this format on your judgment template.")

        if not bool(re.search(r'\{response_b\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_b` format. Please ensure this format on your judgment template.")

    end = False
    while not end:
        end = True
        current_round_contexts = []
        current_round_responses = []
        context_indices = []
        next_round_responses = []
        
        for context_index, (context, remained_responses) in enumerate(zip(contexts, responses)):
            if len(remained_responses) > 1:
                random.shuffle(remained_responses)
                if len(remained_responses) % 2 == 1:
                    next_round_responses.append([remained_responses[0]])
                    remained_responses = remained_responses[1:]
                else:
                     next_round_responses.append([])
                
                for index in range(len(remained_responses)//2):
                    current_round_contexts.append(context)
                    current_round_responses.append(remained_responses[2*index:2*(index+1)])
                    context_indices.append(context_index)
            
            else:
                next_round_responses.append(remained_responses)

        winners, _ = get_batch_judgments(model,
                                         tokenizer,
                                         judge_templates,
                                         token_id_a,
                                         token_id_b,
                                         current_round_contexts,
                                         current_round_responses,
                                         max_length,
                                         judge_batch_size,)
        
        for context_index, winner in zip(context_indices, winners):
            next_round_responses[context_index].append(winner)
            
        responses = next_round_responses
        for remained_responses in responses:
            if len(remained_responses) > 1:
                end = False
                break
        
    responses = [winner[0] for winner in responses]
            
    return responses