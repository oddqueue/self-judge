import re
import random

from typing import Dict, List, Optional

import numpy as np

from transformers import PreTrainedTokenizerBase
from datasets import Dataset

from self_judge import get_conditional_labels


def preprocess_hh_dataset(dataset: Dataset,
                          system_message: Optional[str]='',
                          num_proc: Optional[int]=8):
    
    hh_template = r"(\n\nHuman:.*?)(?=\n\nAssistant:|$)|" r"(\n\nAssistant:.*?)(?=\n\nHuman:|$)"
    user_template = "Human:"
    assistant_template = "Assistant:"
    
    def preprocess(example: Dict[str, str]):
        chosen = example['chosen']
        chosen_splits = re.findall(hh_template, chosen, re.DOTALL)
        chosen_splits = ["".join(chosen_split).strip() for chosen_split in chosen_splits if any(chosen_split)]

        while not chosen_splits[-1].startswith(assistant_template):
            chosen_splits = chosen_splits[:-1]
        
        chosen_messages = [{"role": "system", "content": system_message}]
        for split in chosen_splits:
            if split.startswith(user_template):
                content = split.replace(user_template, "").replace(assistant_template, "").strip()
                message = {"role": "user", "content": content}
                chosen_messages.append(message)

            elif split.startswith(assistant_template):
                content = split.replace(user_template, "").replace(assistant_template, "").strip()
                message = {"role": "assistant", "content": content}
                chosen_messages.append(message)
                
        example['chosen'] = chosen_messages

        rejected = example['rejected']
        rejected_splits = re.findall(hh_template, rejected, re.DOTALL)
        rejected_splits = ["".join(rejected_split).strip() for rejected_split in rejected_splits if any(rejected_split)]

        while not rejected_splits[-1].startswith(assistant_template):
            rejected_splits = rejected_splits[:-1]

        rejected_messages = [{"role": "system", "content": system_message}]
        for split in rejected_splits:
            if split.startswith(user_template):
                content = split.replace(user_template, "").replace(assistant_template, "").strip()
                message = {"role": "user", "content": content}
                rejected_messages.append(message)

            elif split.startswith(assistant_template):
                content = split.replace(user_template, "").replace(assistant_template, "").strip()
                message = {"role": "assistant", "content": content}
                rejected_messages.append(message)

        example['rejected'] = rejected_messages

        return example
    
    dataset = dataset.map(preprocess,
                          batched=False,
                          num_proc=num_proc,)
    
    return dataset


def construct_hh_sft_dataset(dataset: Dataset,
                             tokenizer: PreTrainedTokenizerBase,
                             user_template_ids: List[int],
                             assistant_template_ids: List[int],
                             ignore_index: Optional[int]=-100,
                             num_proc: Optional[int]=8,):

    def preprocess(example: Dict[str, str]):
        messages_ids = tokenizer.apply_chat_template(example['chosen'])
        messages_labels, _ = get_conditional_labels(messages_ids.copy(),
                                                    user_template_ids=user_template_ids,
                                                    assistant_template_ids=assistant_template_ids,
                                                    ignore_index=ignore_index,)

        example['input_ids'] = messages_ids
        example['labels'] = messages_labels
        
        return example
        
    dataset = dataset.map(preprocess,
                          batched=False,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset


def construct_hh_judge_dataset(dataset: Dataset,
                               tokenizer: PreTrainedTokenizerBase,
                               user_template_ids: List[int],
                               assistant_template_ids: List[int],
                               judge_templates: Dict[str, Dict[str, str]],
                               ignore_index: Optional[int] = -100,
                               num_proc: Optional[int] = 8,):

    for principle, judge_template in judge_templates.items():
        if not bool(re.search(r'\{context\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `context` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_a\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_a` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_b\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_b` format. Please insure this format on your judgment template.")
    
    def preprocess(examples: Dict[str, List[str]]):
        input_ids = []
        labels = []
          
        for chosen, rejected in zip(examples['chosen'], examples['rejected']):
            context = ""

            for message in chosen[:-1]:
                content = message['content']

                if message['role'] == 'user':
                    context += "User: " + content + '\n\n'

                elif message['role'] == 'assistant':
                    context += "Assistant: " + content + '\n\n'

            context = context.strip()
            
            for principle, judge_template in judge_templates.items():
                judge_system = judge_template['system']
                
                judge_user_a = judge_template['user'].format(context=context,
                                                             response_a=chosen[-1]['content'],
                                                             response_b=rejected[-1]['content'],)
                judge_user_b = judge_template['user'].format(context=context,
                                                             response_a=rejected[-1]['content'],
                                                             response_b=chosen[-1]['content'],)

                judge_assistant_a = judge_template['assistant'] + " A"
                judge_assistant_b = judge_template['assistant'] + " B"

                judge_a = [{"role": "system", "content": judge_system},
                           {"role": "user", "content": judge_user_a},
                           {"role": "assistant", "content": judge_assistant_a}]
                judge_b = [{"role": "system", "content": judge_system},
                           {"role": "user", "content": judge_user_b},
                           {"role": "assistant", "content": judge_assistant_b}]

                judge_a = tokenizer.apply_chat_template(judge_a, tokenize=False)
                judge_b = tokenizer.apply_chat_template(judge_b, tokenize=False)

                if judge_a.endswith(tokenizer.eos_token):
                    judge_a = judge_a[:-len(tokenizer.eos_token)]
                if judge_b.endswith(tokenizer.eos_token):
                    judge_b = judge_b[:-len(tokenizer.eos_token)]

                judge_a_ids = tokenizer.encode(judge_a)
                judge_b_ids = tokenizer.encode(judge_b)

                judge_a_labels, _ = get_conditional_labels(judge_a_ids.copy(),
                                                           user_template_ids=user_template_ids,
                                                           assistant_template_ids=assistant_template_ids,
                                                           ignore_index=ignore_index,)     

                judge_b_labels, _ = get_conditional_labels(judge_b_ids.copy(),
                                                           user_template_ids=user_template_ids,
                                                           assistant_template_ids=assistant_template_ids,
                                                           ignore_index=ignore_index,)
                    
                input_ids.append(judge_a_ids)
                input_ids.append(judge_b_ids)
                labels.append(judge_a_labels)
                labels.append(judge_b_labels)

        examples['input_ids'] = input_ids
        examples['labels'] = labels
        
        return examples
    
    dataset = dataset.map(preprocess,
                          batched=True,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset


def construct_hh_self_judge_dataset(dataset: Dataset,
                                    tokenizer: PreTrainedTokenizerBase,
                                    num_proc: Optional[int] = 8,):

    def preprocess(example: Dict[str, str]):
        chosen = example['chosen']
        rejected = example['rejected']
        
        query = tokenizer.apply_chat_template(chosen[:-1], tokenize=False, add_generation_prompt=True)
        context = ""
        
        for message in chosen[:-1]:
            content = message['content']

            if message['role'] == 'user':
                context += "User: " + content + '\n\n'

            elif message['role'] == 'assistant':
                context += "Assistant: " + content + '\n\n'

        context = context.strip()

        example['query'] = query
        example['context'] = context
        example['chosen'] = tokenizer.apply_chat_template(chosen, tokenize=False)
        example['rejected'] = tokenizer.apply_chat_template(rejected, tokenize=False)
        
        return example
    
    dataset = dataset.map(preprocess,
                          batched=False,
                          num_proc=num_proc,)
    
    return dataset


def construct_uf_sft_dataset(dataset: Dataset,
                             tokenizer: PreTrainedTokenizerBase,
                             user_template_ids: List[int],
                             assistant_template_ids: List[int],
                             ignore_index: Optional[int] = -100,
                             num_proc: Optional[int] = 8,):
    
    def preprocess(examples: Dict[str, List[str]]):
        input_ids = []
        labels = []

        for instruction, completions in zip(examples['instruction'], examples['completions']):
            if len(completions) == 0:
                continue

            system_messages = []
            
            responses = []
            response_lengths = {}
            response_ratings = {index: [] for index in range(len(completions))}

            for response_index, completion in enumerate(completions):
                system_messages.append(completion['custom_system_prompt'])
                
                response = completion['response']
                responses.append(response)
                response_lengths[response_index] = len(response)

                for principle, annotation in completion['annotations'].items():
                    if annotation['Rating'] != "N/A":
                        rating = float(annotation['Rating'])
                        response_ratings[response_index].append(rating)

            response_ratings = {index: np.mean(ratings) for index, ratings in response_ratings.items()}
            response_rank = [index for index, _ in sorted(response_ratings.items(), key=lambda item: (item[1], response_lengths[item[0]]), reverse=True)]
            
            best_system_message = system_messages[response_rank[0]]
            best_response = responses[response_rank[0]]
            
            messages = [{"role": "system", "content": best_system_message},
                        {"role": "user", "content": instruction},
                        {"role": "assistant", "content": best_response}]
            
            messages_ids = tokenizer.apply_chat_template(messages)
            messages_labels, _ = get_conditional_labels(messages_ids.copy(),
                                                        user_template_ids=user_template_ids,
                                                        assistant_template_ids=assistant_template_ids,
                                                        ignore_index=ignore_index,)    

            input_ids.append(messages_ids)
            labels.append(messages_labels)

        examples['input_ids'] = input_ids
        examples['labels'] = labels

        return examples
    
    dataset = dataset.map(preprocess,
                          batched=True,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset


def construct_uf_judge_no_rationale_dataset(dataset: Dataset,
                                            tokenizer: PreTrainedTokenizerBase,
                                            user_template_ids: List[str],
                                            assistant_template_ids: List[str],
                                            judge_templates: Dict[str, Dict[str, str]],
                                            ignore_index: Optional[int] = -100,
                                            num_proc: Optional[int] = 8,):
    
    for principle, judge_template in judge_templates.items():
        if not bool(re.search(r'\{context\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `context` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_a\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_a` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_b\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_b` format. Please insure this format on your judgment template.")
    
    def preprocess(examples: Dict[str, List[str]]):
        input_ids = []
        labels = []

        for instruction, completions in zip(examples['instruction'], examples['completions']):
            if len(completions) == 0:
                continue
            
            responses = []
            response_lengths = {}
            response_ratings = {index: [] for index in range(len(completions))}

            for response_index, completion in enumerate(completions):
                response = completion['response']
                responses.append(response)
                response_lengths[response_index] = len(response)
                
                for principle, annotation in completion['annotations'].items():
                    if annotation['Rating'] != "N/A":
                        rating = float(annotation['Rating'])
                        response_ratings[response_index].append(rating)

            response_ratings = {index: np.mean(ratings) for index, ratings in response_ratings.items()}
            response_rank = [index for index, _ in sorted(response_ratings.items(), key=lambda item: (item[1], response_lengths[item[0]]), reverse=True)]
            
            comparisons = []
            target_response_index = response_rank[0]
            for comparison_response_index in response_rank[1:]:
                if response_ratings[target_response_index] > response_ratings[comparison_response_index]:
                    comparisons.append((target_response_index, comparison_response_index))

            if len(comparisons) == 0:
                continue
                
            chosen_index, rejected_index = random.sample(comparisons, 1)[0]
            
            for principle, judge_template in judge_templates.items():
                judge_system = judge_template['system']
                
                chosen = responses[chosen_index]
                rejected = responses[rejected_index]

                judge_user_a = judge_template['user'].format(context=instruction,
                                                             response_a=chosen,
                                                             response_b=rejected,)

                judge_user_b = judge_template['user'].format(context=instruction,
                                                             response_a=rejected,
                                                             response_b=chosen,)

                judge_assistant_a = judge_template['assistant'] + " A"
                judge_assistant_b = judge_template['assistant'] + " B"

                judge_a = [{"role": "system", "content": judge_system},
                           {"role": "user", "content": judge_user_a},
                           {"role": "assistant", "content": judge_assistant_a}]

                judge_b = [{"role": "system", "content": judge_system},
                           {"role": "user", "content": judge_user_b},
                        {"role": "assistant", "content": judge_assistant_b}]

                judge_a = tokenizer.apply_chat_template(judge_a, tokenize=False)
                judge_b = tokenizer.apply_chat_template(judge_b, tokenize=False)

                if judge_a.endswith(tokenizer.eos_token):
                    judge_a = judge_a[:-len(tokenizer.eos_token)]
                if judge_b.endswith(tokenizer.eos_token):
                    judge_b = judge_b[:-len(tokenizer.eos_token)]

                judge_a_ids = tokenizer.encode(judge_a)
                judge_b_ids = tokenizer.encode(judge_b)

                judge_a_labels, _ = get_conditional_labels(judge_a_ids.copy(),
                                                           user_template_ids=user_template_ids,
                                                           assistant_template_ids=assistant_template_ids,
                                                           ignore_index=ignore_index,)    

                judge_b_labels, _ = get_conditional_labels(judge_b_ids.copy(),
                                                           user_template_ids=user_template_ids,
                                                           assistant_template_ids=assistant_template_ids,
                                                           ignore_index=ignore_index,)

                input_ids.append(judge_a_ids)
                input_ids.append(judge_b_ids)

                labels.append(judge_a_labels)
                labels.append(judge_b_labels)
                    
        examples['input_ids'] = input_ids
        examples['labels'] = labels

        return examples
    
    dataset = dataset.map(preprocess,
                          batched=True,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset


def construct_uf_judge_rationale_dataset(dataset: Dataset,
                                         tokenizer: PreTrainedTokenizerBase,
                                         user_template_ids: List[int],
                                         assistant_template_ids: List[int],
                                         judge_templates: Dict[str, Dict[str, str]],
                                         decision_rationale_templates: Dict[str, str],
                                         ignore_index: Optional[int] = -100,
                                         num_proc: Optional[int] = 8,):
    
    principles = ["helpfulness", "honesty", "instruction_following", "truthfulness"]
    
    for principle, judge_template in judge_templates.items():
        if not bool(re.search(r'\{context\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `context` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_a\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_a` format. Please insure this format on your judgment template.")

        if not bool(re.search(r'\{response_b\}', judge_template['user'])):
            raise ValueError(f"Your `judge_templates` for `user` on `{principle}` does not have `response_b` format. Please insure this format on your judgment template.")
    
    def preprocess(examples: List[Dict[str, str]]):
        input_ids = []
        labels = []

        for instruction, completions in zip(examples['instruction'], examples['completions']):
            if len(completions) == 0:
                continue
            
            responses = []
            response_lengths = {}
            response_ratings = {index: [] for index in range(len(completions))}
            
            principle_rationales = {principle: [] for principle in principles}
            principle_ratings = {principle: [] for principle in principles}

            for response_index, completion in enumerate(completions):
                response = completion['response']
                responses.append(response)
                response_lengths[response_index] = len(response)
                
                for principle, annotation in completion['annotations'].items():
                    rationale = annotation['Rationale']
                    if 'Rationale For Rating' in annotation.keys():
                        rationale += " " + annotation['Rationale For Rating']
                    principle_rationales[principle].append(rationale)

                    if annotation['Rating'] != "N/A":
                        rating = float(annotation['Rating'])
                        principle_ratings[principle].append(rating)
                        response_ratings[response_index].append(rating)
                    else:
                        principle_ratings[principle].append("N/A")

            response_ratings = {index: np.mean(ratings) for index, ratings in response_ratings.items()}
            response_rank = [index for index, _ in sorted(response_ratings.items(), key=lambda item: (item[1], response_lengths[item[0]]), reverse=True)]

            principle_comparisons = {principle: [] for principle in principles}
            for principle, principle_rating in principle_ratings.items():
                comparisons = []
                for target_response_index in response_rank:
                    for comparison_response_index in range(len(response_rank)):
                        if principle_rating[target_response_index] != "N/A" and principle_rating[comparison_response_index] != "N/A":
                            if principle_rating[target_response_index] > principle_rating[comparison_response_index]:
                                comparison = (target_response_index, comparison_response_index)
                            elif principle_rating[comparison_response_index] > principle_rating[target_response_index]:
                                comparison = (comparison_response_index, target_response_index)
                            else:
                                comparison = ()

                            if len(comparison) != 0 and "Text" not in principle_rationales[principle][comparison[0]] and "Text" not in principle_rationales[principle][comparison[1]]:
                                comparisons.append(comparison)

                    if len(comparisons) != 0:
                        comparisons = random.sample(comparisons, 1)
                        break

                principle_comparisons[principle] = comparisons

            for principle, comparisons in principle_comparisons.items():
                judge_template = judge_templates[principle]

                for chosen_index, rejected_index in comparisons:
                    chosen = responses[chosen_index]
                    chosen_rationale = principle_rationales[principle][chosen_index]

                    rejected = responses[rejected_index]
                    rejected_rationale = principle_rationales[principle][rejected_index]

                    judge_system = judge_template['system']
                
                    judge_user_a = judge_template['user'].format(context=instruction,
                                                                 response_a=chosen,
                                                                 response_b=rejected,)
                    
                    judge_user_b = judge_template['user'].format(context=instruction,
                                                                 response_a=rejected,
                                                                 response_b=chosen,)

                    judge_assistant_a = judge_template['assistant']\
                                        + decision_rationale_templates['decision'] + " A" + '\n'\
                                        + decision_rationale_templates['rationale'].format(position="A") + " " + chosen_rationale\
                                        + decision_rationale_templates['rationale'].format(position="B") + " " + rejected_rationale

                    judge_assistant_b = judge_template['assistant']\
                                        + decision_rationale_templates['decision'] + " B" + '\n'\
                                        + decision_rationale_templates['rationale'].format(position="A") + " " + rejected_rationale\
                                        + decision_rationale_templates['rationale'].format(position="B") + " " + chosen_rationale

                    judge_a = [{"role": "system", "content": judge_system},
                               {"role": "user", "content": judge_user_a},
                               {"role": "assistant", "content": judge_assistant_a}]

                    judge_b = [{"role": "system", "content": judge_system},
                               {"role": "user", "content": judge_user_b},
                               {"role": "assistant", "content": judge_assistant_b}]

                    judge_a = tokenizer.apply_chat_template(judge_a, tokenize=False)
                    judge_b = tokenizer.apply_chat_template(judge_b, tokenize=False)

                    if judge_a.endswith(tokenizer.eos_token):
                        judge_a = judge_a[:-len(tokenizer.eos_token)]
                    if judge_b.endswith(tokenizer.eos_token):
                        judge_b = judge_b[:-len(tokenizer.eos_token)]

                    judge_a_ids = tokenizer.encode(judge_a)
                    judge_b_ids = tokenizer.encode(judge_b)

                    judge_a_labels, _ = get_conditional_labels(judge_a_ids.copy(),
                                                               user_template_ids=user_template_ids,
                                                               assistant_template_ids=assistant_template_ids,
                                                               ignore_index=ignore_index,)    
                    
                    judge_b_labels, _ = get_conditional_labels(judge_b_ids.copy(),
                                                               user_template_ids=user_template_ids,
                                                               assistant_template_ids=assistant_template_ids,
                                                               ignore_index=ignore_index,)

                    input_ids.append(judge_a_ids)
                    input_ids.append(judge_b_ids)

                    labels.append(judge_a_labels)
                    labels.append(judge_b_labels)

        examples['input_ids'] = input_ids
        examples['labels'] = labels

        return examples
    
    dataset = dataset.map(preprocess,
                          batched=True,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset


def construct_uf_self_judge_dataset(dataset: Dataset,
                                    tokenizer: PreTrainedTokenizerBase,
                                    num_proc: Optional[int] = 8,):
    
    def preprocess(examples: List[Dict[str, str]]):
        queries = []
        contexts = []
        chosens = []
        rejecteds = []

        for instruction, completions in zip(examples['instruction'], examples['completions']):
            if len(completions) == 0:
                continue

            system_messages = []
            
            responses = []
            response_lengths = {}
            response_ratings = {index: [] for index in range(len(completions))}

            for response_index, completion in enumerate(completions):
                system_messages.append(completion['custom_system_prompt'])
                
                response = completion['response']
                responses.append(response)
                response_lengths[response_index] = len(response)
                
                for principle, annotation in completion['annotations'].items():
                    if annotation['Rating'] != "N/A":
                        rating = float(annotation['Rating'])
                        response_ratings[response_index].append(rating)

            response_ratings = {index: np.mean(ratings) for index, ratings in response_ratings.items()}
            response_rank = [index for index, _ in sorted(response_ratings.items(), key=lambda item: (item[1], response_lengths[item[0]]), reverse=True)]
            
            messages = [{"role": "system", "content": system_messages[response_rank[0]]},
                        {"role": "user", "content": instruction}]
            query = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            chosen = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": responses[response_rank[0]]}], tokenize=False)
            rejected = tokenizer.apply_chat_template(messages + [{"role": "assistant", "content": responses[response_rank[-1]]}], tokenize=False)
            
            queries.append(query)
            contexts.append(instruction)
            chosens.append(chosen)
            rejecteds.append(rejected)

        examples['query'] = queries
        examples['context'] = contexts
        examples['chosen'] = chosens
        examples['rejected'] = rejecteds

        return examples
    
    dataset = dataset.map(preprocess,
                          batched=True,
                          num_proc=num_proc,
                          remove_columns=dataset.column_names)
    
    return dataset