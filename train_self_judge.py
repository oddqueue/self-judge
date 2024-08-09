import argparse
from omegaconf import OmegaConf

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from datasets import load_dataset
from peft import LoraConfig, PeftModel
from trl import set_seed

from self_judge import SelfJudgeCollator, SelfJudgeTrainer
from preprocess import(
    preprocess_hh_dataset,
    construct_hh_self_judge_dataset,
    construct_uf_self_judge_dataset,
)
                                        
def main(config):
    set_seed(config.training_args.seed)
    training_args = TrainingArguments(**config.training_args)
    
    model = AutoModelForCausalLM.from_pretrained(config.model.pretrained_model_name_or_path,
                                                 attn_implementation=config.model.attn_implementation,
                                                 torch_dtype=torch.bfloat16)
    
    if "peft_model_name_or_path" in config.model:
        model = PeftModel.from_pretrained(model, config.model.peft_model_name_or_path, is_trainable=True)
        model = model.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(config.model.peft_model_name_or_path)  
        
    else:     
        tokenizer = AutoTokenizer.from_pretrained(config.model.pretrained_model_name_or_path)  
    
    if "peft_config" in config:
        peft_config = LoraConfig(**config.peft_config,
                                inference_mode=False,
                                bias="none",
                                task_type="CAUSAL_LM")
    else:
        peft_config = None
    
    if config.dataset == "Anthropic/hh-rlhf":
        
        dataset = load_dataset(config.dataset, data_dir="helpful-base", split='train')
        dataset = preprocess_hh_dataset(dataset, config.templates.judge.helpful.system)
        
        train_dataset = construct_hh_self_judge_dataset(dataset, tokenizer)
        
        judge_templates = config.templates.judge
    
    elif config.dataset == "openbmb/UltraFeedback":
        dataset = load_dataset(config.dataset, split='train')
        
        train_dataset = construct_uf_self_judge_dataset(dataset, tokenizer)
        
        if "decision_rationale" in config.templates:
            judge_templates = config.templates.judge.copy()
            for principle, judge_template in judge_templates.items():
                judge_template.assistant = judge_template.assistant + config.templates.decision_rationale.decision

        else:
            judge_templates = config.templates.judge

    else:
        raise ValueError(
                f"Unknown dataset: {config.dataset}. Should be one of ['Anthropic/hh-rlhf', 'openbmb/UltraFeedback']"
            )

    data_collator = SelfJudgeCollator()

    trainer = SelfJudgeTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            args=training_args,
            train_dataset=train_dataset,
            judge_templates=judge_templates,
            peft_config=peft_config,
            **config.self_judge_args,
        )
    
    trainer.train()

    trainer.save_model(config.training_args.output_dir)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SELF-JUDGE configuration')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='path to configuration',
    )
    
    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    main(config)