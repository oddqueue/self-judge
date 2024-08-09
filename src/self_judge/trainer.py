import warnings
from typing import Any, Dict, List, Tuple, Optional, Union, Literal
from collections import defaultdict
from copy import deepcopy

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from trl import create_reference_model
from peft import PeftConfig, get_peft_model

import deepspeed

from .utils import (
    get_conditional_labels,
    get_batch_judgments,
    gradient_checkpointing_context,
    PeftSavingCallback
)

    
class SelfJudgeTrainer(Trainer):
    def __init__(self,
                 model: Union[PreTrainedModel, nn.Module],
                 tokenizer: PreTrainedTokenizerBase,
                 args: TrainingArguments,
                 data_collator: DataCollator,
                 train_dataset: Dataset,
                 judge_templates: Dict[str, Dict[str, str]],
                 token_id_a: int,
                 token_id_b: int,
                 ref_model: Optional[Union[PreTrainedModel, nn.Module]] = None,
                 peft_config: Optional[Dict] = None,
                 eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
                 callbacks: Optional[List[TrainerCallback]] = None,
                 optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
                 disable_dropout: Optional[bool] = True,
                 max_length: Optional[int] = 2048,
                 max_new_tokens: Optional[int] = 768,
                 judge_batch_size: Optional[int] = 1,
                 top_p: Optional[float] = 0.9,
                 temperature: Optional[float] = 1.0,
                 loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid",
                 beta: Optional[float] = 0.1,
                 average_log_prob: Optional[bool] = False,
                 ignore_index: Optional[int] = -100,):
       
        # Case 1; reference model was given
        if ref_model is not None:
            self.ref_model = ref_model
        
        # Case 2; policy model is PeftModel
        elif peft_config is not None:
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:
                def make_inputs_require_grad(module, input, output):
                     output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
            
            if not isinstance(peft_config, PeftConfig):
                raise ValueError(
                    "If you want to use the PeftModel, you need to pass a PeftConfig object to the SelfJudgeTrainer."
                    f" and you passed a {type(peft_config)}."
                )
            
            model = get_peft_model(model, peft_config)
            
            if callbacks is None:
                callbacks = [PeftSavingCallback]
                
            self.ref_model = None

        # Case 3; reference model is same as initial policy
        else:
            self.ref_model = create_reference_model(model)
        
        if args.remove_unused_columns:
            args.remove_unused_columns = False
            warnings.warn(
                "When using `SelfJudgeTrainer`, you should set `remove_unused_columns=False` in your TrainingArguments"
                " we have set it for you, but you should do it yourself in the future.",
            )
            
        if disable_dropout:
            for module in model.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0

            if self.ref_model is not None:
                for module in self.ref_model.modules():
                    if isinstance(module, torch.nn.Dropout):
                        module.p = 0
        
        self.max_length = max_length
        self.max_new_tokens = max_new_tokens
        self.top_p = top_p
        self.temperature = temperature

        self.judge_templates = judge_templates
        self.token_id_a = token_id_a
        self.token_id_b = token_id_b
        self.ignore_index = ignore_index
        self.judge_batch_size = judge_batch_size

        self.loss_type = loss_type
        self.beta = beta
        self.average_log_prob = average_log_prob

        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )
            
        if self.ref_model is None:
            if not hasattr(self.accelerator.unwrap_model(self.model), "disable_adapter") or not hasattr(self.accelerator.unwrap_model(self.model), "set_adapter"):
                raise ValueError(
                    "You are using a `peft` version that does not support `disable_adapter` or `set_adapter`. Please update your `peft` version to the latest version."
                )
        else:
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
            
            
    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
        ):

        loss, metrics = self.get_batch_metrics(model, inputs, train_eval="train")
        
        if self.accelerator.is_main_process:
            if metrics is not None:
                self.store_metrics(metrics, train_eval="train")
        
        if return_outputs:
            return (loss, metrics)
        return loss
    
        
    def get_batch_metrics(self,
                          model: Union[PreTrainedModel, nn.Module],
                          inputs: Dict[str, Union[torch.Tensor, Any]],
                          train_eval: Literal["train", "eval"] = "train",):  
        
        if train_eval == "train":
            inputs = self.get_batch_responses(self.accelerator.unwrap_model(self.model), inputs)
            
        inputs, labels, stats = self.get_batch_inputs(inputs)

        with torch.no_grad():
             # Case; separated reference model
            if self.ref_model:
                reference_logits = self.accelerator.unwrap_model(self.ref_model)(**inputs).logits
                reference_logps = self.get_batch_logps(reference_logits, labels, average_log_prob=self.average_log_prob)
            # Case; policy model is PeftModel from reference model
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    reference_logits = self.model(**inputs).logits
                    reference_logps = self.get_batch_logps(reference_logits, labels, average_log_prob=self.average_log_prob)

        policy_logits = model(**inputs).logits 
        policy_logps = self.get_batch_logps(policy_logits, labels, average_log_prob=self.average_log_prob)
        
        policy_chosen_logits, policy_rejected_logits = torch.chunk(policy_logits, 2)
        reference_chosen_logits, reference_rejected_logits = torch.chunk(reference_logits, 2)

        policy_chosen_logps, policy_rejected_logps = torch.chunk(policy_logps, 2)
        reference_chosen_logps, reference_rejected_logps = torch.chunk(reference_logps, 2)

        losses, chosen_rewards, rejected_rewards = self.get_dpo_loss(policy_chosen_logps,
                                                                     policy_rejected_logps,
                                                                     reference_chosen_logps,
                                                                     reference_rejected_logps,)

        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        prefix = "eval_" if train_eval == "eval" else ""
        metrics = {}
        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.cpu().numpy().mean()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.cpu().numpy().mean()
        metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).cpu().numpy().mean()
        metrics[f"{prefix}logps/policy_chosen"] = policy_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/policy_rejected"] = policy_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/reference_chosen"] = reference_chosen_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logps/reference_rejected"] = reference_rejected_logps.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/policy_chosen"] = policy_chosen_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/policy_rejected"] = policy_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/reference_chosen"] = reference_chosen_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}logits/reference_rejected"] = reference_rejected_logits.detach().cpu().numpy().mean()
        metrics[f"{prefix}tokens/chosen_length_mean"] = stats['tokens_chosen_length_mean']
        metrics[f"{prefix}tokens/chosen_length_std"] = stats['tokens_chosen_length_std']
        metrics[f"{prefix}tokens/rejected_length_mean"] = stats['tokens_rejected_length_mean']
        metrics[f"{prefix}tokens/rejected_length_std"] = stats['tokens_rejected_length_std']

        return losses.mean(), metrics
    
    
    def get_batch_responses(self,
                            model: Union[PreTrainedModel, nn.Module],
                            inputs: Dict[str, List[str]],):
       
        with gradient_checkpointing_context(model, model.is_gradient_checkpointing): 
            encoded_inputs = self.tokenizer(inputs['query'],
                                            padding=True,
                                            truncation=True,
                                            max_length=self.max_length,
                                            return_tensors='pt').to(self.accelerator.device)
            
            outputs = self.model.generate(
                **encoded_inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                top_p=self.top_p,
                temperature=self.temperature,
                pad_token_id=self.tokenizer.pad_token_id,
                num_return_sequences=2,
                use_cache=True,
            )
            
        decoded_outputs = self.tokenizer.batch_decode(outputs[:, encoded_inputs.input_ids.shape[1]:], skip_special_tokens=True)
        responses = [decoded_outputs[context_start_index:context_start_index+2] for context_start_index in range(0, len(decoded_outputs), 2)]
        
        # Case; reference model as a judge
        if self.ref_model:
            chosen, rejected = get_batch_judgments(self.accelerator.unwrap_model(self.ref_model),
                                                   self.tokenizer,
                                                   self.judge_templates,
                                                   self.token_id_a,
                                                   self.token_id_b,
                                                   inputs['context'],
                                                   responses,
                                                   self.max_length,
                                                   self.judge_batch_size,)
        # Case; policy model is PeftModel from reference model
        else:
            with self.accelerator.unwrap_model(self.model).disable_adapter():
                chosen, rejected = get_batch_judgments(self.model,
                                                       self.tokenizer,
                                                       self.judge_templates,
                                                       self.token_id_a,
                                                       self.token_id_b,
                                                       inputs['context'],
                                                       responses,
                                                       self.max_length,                                                      
                                                       self.judge_batch_size,)
                
        chosen = [query + response + self.tokenizer.eos_token for query, response in zip(inputs['query'], chosen)]
        rejected = [query + response + self.tokenizer.eos_token for query, response in zip(inputs['query'], rejected)]
        
        inputs['chosen'] = chosen
        inputs['rejected'] = rejected
            
        return inputs
        
        
    def get_batch_inputs(self, inputs: Dict[str, List[Union[str, List[str]]]]):

        batch = self.tokenizer(inputs['chosen'] + inputs['rejected'],
                               padding=True,
                               truncation=True,
                               max_length=self.max_length,
                               return_tensors='pt')
        batch_chosen_input_ids, batch_rejected_input_ids = batch.input_ids.clone().chunk(2)
        
        batch_query_input_ids = self.tokenizer(inputs['query'], max_length=self.max_length).input_ids
        
        batch_chosen_labels = []
        batch_rejected_labels = []
        chosen_lengths = []
        rejected_lengths = []
        
        for query_input_ids, chosen_input_ids, rejected_input_ids in zip(batch_query_input_ids, batch_chosen_input_ids, batch_rejected_input_ids):
            chosen_labels, chosen_length = get_conditional_labels(chosen_input_ids, None, query_input_ids, self.ignore_index, True)
            rejected_labels, rejected_length = get_conditional_labels(rejected_input_ids, None, query_input_ids, self.ignore_index, True)
        
            batch_chosen_labels.append(chosen_labels)
            batch_rejected_labels.append(rejected_labels)
            chosen_lengths.append(chosen_length)
            rejected_lengths.append(rejected_length)
        
        batch = batch.to(self.accelerator.device)
        labels = torch.vstack([torch.vstack(batch_chosen_labels), torch.vstack(batch_rejected_labels)]).to(self.accelerator.device)
            
        stats = {'tokens_chosen_length_mean': np.nanmean(chosen_lengths),
                 'tokens_chosen_length_std': np.nanstd(chosen_lengths),
                 'tokens_rejected_length_mean': np.nanmean(rejected_lengths),
                 'tokens_rejected_length_std': np.nanstd(rejected_lengths)}
                                    
        return batch, labels, stats
    
        
    def get_batch_logps(self,
                        logits: torch.Tensor,
                        labels: torch.Tensor,
                        average_log_prob: Optional[bool] = False,):
        
        """Compute the log probabilities of the given labels under the given logits.

        Args:
            logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
            labels: Labels for which to compute the log probabilities. Label tokens with a value of label_pad_token_id are ignored. Shape: (batch_size, sequence_length)
            average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

        Returns:
            A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
        """
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = (labels != self.ignore_index)

        # dummy token; we'll ignore the losses on these tokens later
        labels[labels == self.ignore_index] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    
    def get_dpo_loss(self,
                     policy_chosen_logps: torch.Tensor,
                     policy_rejected_logps: torch.Tensor,
                     reference_chosen_logps: torch.Tensor,
                     reference_rejected_logps: torch.Tensor,):

        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios
        
        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            # eqn (17) of the paper where beta is the regularization parameter for the IPO loss, denoted by tau in the paper.
            losses = (logits - 1 / (2 * self.beta)) ** 2
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo']"
            )
        
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        return losses, chosen_rewards, rejected_rewards
    
  
    def prediction_step(self,
                        model: Union[PreTrainedModel, nn.Module],
                        inputs: Dict[str, Union[torch.Tensor, Any]],
                        prediction_loss_only: bool,
                        ignore_keys: Optional[List[str]] = None,):

        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        with torch.no_grad():
            loss, metrics = self.get_batch_metrics(model, inputs, train_eval="eval")

        if loss is not None:
            loss = loss.detach()
            
        # force log the metrics
        if self.accelerator.is_main_process:
            if metrics is not None:
                self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss, None, None)

        # logits for the chosen and rejected samples from model
        logits_dict = {
            "eval_logits/policy_chosen": metrics["eval_logits/policy_chosen"],
            "eval_logits/policy_rejected": metrics["eval_logits/policy_rejected"],
            "eval_logits/reference_chosen": metrics["eval_logits/reference_chosen"],
            "eval_logits/reference_rejected": metrics["eval_logits/reference_rejected"],
        }
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1)
        labels = torch.zeros(logits.shape[0])

        return (loss.detach(), logits, labels)
    
    
    def store_metrics(self,
                      metrics: Dict[str, float],
                      train_eval: Literal["train", "eval"] = "train",):
        
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)


    def log(self, logs: Dict[str, float]):
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)
    
    
    def _prepare_deepspeed(self, model: PreTrainedModel):
        # Adapted from accelerate: https://github.com/huggingface/accelerate/blob/739b135f8367becb67ffaada12fe76e3aa60fefd/src/accelerate/accelerator.py#L1473
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    # Note that `stage3_prefetch_bucket_size` can produce DeepSpeed messages like: `Invalidate trace cache @ step 0: expected module 1, but got module 0`
                    # This is expected and is not an error, see: https://github.com/microsoft/DeepSpeed/discussions/4081
                    config_kwargs.update(
                        {
                            "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                            "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                            "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        # If ZeRO-3 is used, we shard both the active and reference model.
        # Otherwise, we assume the reference model fits in memory and is initialized on each device with ZeRO disabled (stage 0)
        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model