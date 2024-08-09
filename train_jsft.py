import argparse
from omegaconf import OmegaConf

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, PeftConfig, get_peft_model
from trl import set_seed

from self_judge import JudgeAugmentedSFTCollator, PeftSavingCallback
from preprocess import(
    preprocess_hh_dataset,
    construct_hh_sft_dataset,
    construct_hh_judge_dataset,
    construct_uf_sft_dataset,
    construct_uf_judge_no_rationale_dataset,
    construct_uf_judge_rationale_dataset,
)
                                        
def main(config):
    set_seed(config.training_args.seed)
    training_args = TrainingArguments(**config.training_args)
    
    model = AutoModelForCausalLM.from_pretrained(config.model.pretrained_model_name_or_path,
                                                 attn_implementation=config.model.attn_implementation,
                                                 torch_dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)
    tokenizer.chat_template = config.templates.chat
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    
    if "peft_config" in config:
        peft_config = LoraConfig(**config.peft_config,
                                 inference_mode=False,
                                 bias="none",
                                 task_type="CAUSAL_LM")
        
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
        callbacks = [PeftSavingCallback]
        
    else:
        callbacks = None
    
    user_template_ids = tokenizer.encode(config.templates.user, add_special_tokens=False)
    assistant_template_ids = tokenizer.encode(config.templates.assistant, add_special_tokens=False)
    
    if config.dataset == "Anthropic/hh-rlhf":
        judge_assistant_template_ids = tokenizer.encode(config.templates.judge.helpful.assistant, add_special_tokens=False)
         
        if config.templates.skip_first_token:
            user_template_ids = user_template_ids[1:]
            assistant_template_ids = assistant_template_ids[1:]
            judge_assistant_template_ids = judge_assistant_template_ids[1:]
        
        base_dataset = load_dataset(config.dataset, data_dir="helpful-base", split='train')
        base_dataset = preprocess_hh_dataset(base_dataset, config.templates.judge.helpful.system)
        online_dataset = load_dataset(config.dataset, data_dir="helpful-online", split='train')
        online_dataset = preprocess_hh_dataset(online_dataset, config.templates.judge.helpful.system)
        
        base_sft_dataset = construct_hh_sft_dataset(base_dataset,
                                                    tokenizer,
                                                    user_template_ids,
                                                    assistant_template_ids)
        online_sft_dataset = construct_hh_sft_dataset(online_dataset,
                                                      tokenizer,
                                                      user_template_ids,
                                                      assistant_template_ids)
        judge_dataset = construct_hh_judge_dataset(base_dataset,
                                                   tokenizer,
                                                   user_template_ids,
                                                   judge_assistant_template_ids,
                                                   config.templates.judge)
        
        train_dataset = concatenate_datasets([base_sft_dataset, online_sft_dataset, judge_dataset])
    
    elif config.dataset == "openbmb/UltraFeedback":
        dataset = load_dataset(config.dataset, split='train')
        
        if not "decision_rationale" in config.templates:
            judge_assistant_template_ids = tokenizer.encode(config.templates.judge.base.assistant, add_special_tokens=False)
            
            if config.templates.skip_first_token:
                user_template_ids = user_template_ids[1:]
                assistant_template_ids = assistant_template_ids[1:]
                judge_assistant_template_ids = judge_assistant_template_ids[1:]
                
            sft_dataset = construct_uf_sft_dataset(dataset,
                                                tokenizer,
                                                user_template_ids,
                                                assistant_template_ids,)
            judge_dataset = construct_uf_judge_no_rationale_dataset(dataset,
                                                                    tokenizer,
                                                                    user_template_ids,
                                                                    judge_assistant_template_ids,
                                                                    config.templates.judge)
        
        else:
            judge_assistant_template_ids = tokenizer.encode(config.templates.decision_rationale.decision, add_special_tokens=False)
            
            if config.templates.skip_first_token:
                user_template_ids = user_template_ids[1:]
                assistant_template_ids = assistant_template_ids[1:]
                judge_assistant_template_ids = judge_assistant_template_ids[1:]

            sft_dataset = construct_uf_sft_dataset(dataset,
                                                tokenizer,
                                                user_template_ids,
                                                assistant_template_ids,)
            judge_dataset = construct_uf_judge_rationale_dataset(dataset,
                                                                 tokenizer,
                                                                 user_template_ids,
                                                                 judge_assistant_template_ids,
                                                                 config.templates.judge,
                                                                 config.templates.decision_rationale)

        train_dataset = concatenate_datasets([sft_dataset, judge_dataset])
        
    else:
        raise ValueError(
                f"Unknown dataset: {config.dataset}. Should be one of ['Anthropic/hh-rlhf', 'openbmb/UltraFeedback']"
            )

    data_collator = JudgeAugmentedSFTCollator(tokenizer=tokenizer,
                                              max_length=config.jsft_args.max_length)
    
    trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=callbacks,
        )
    
    trainer.train()

    trainer.save_model(config.training_args.output_dir)
    

if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description='JSFT configuration')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to configuration',
    )
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)