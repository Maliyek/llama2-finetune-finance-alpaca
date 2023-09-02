!pip install -U transformers datasets 
!pip install -U accelerate
!pip install trl tiktoken peft bitsandbytes 

!pip install huggingface_hub
!python -c "from huggingface_hub.hf_api import HfFolder; HfFolder.save_token('<YOUR API TOKEN>')"

import pandas as pd
from datasets import load_dataset

alpaca_finance = load_dataset("gbharti/finance-alpaca")

alpaca_finance_df = pd.DataFrame(alpaca_finance['train'])
alpaca_finance_df = alpaca_finance_df.fillna("")

text_col=[]

for _,row in alpaca_finance_df.iterrows():
    prompt = "Below is an instruction that describes the task, paired with an input that provides further context.Write a response that appropriately completes the request \n \n"
    instruction = str(row['instruction'])
    input_query = str(row['input'])
    response = str(row['output'])
    
    if len(input_query.strip())==0:
        text = prompt + "### Instruction:\n" +  instruction + "\n###Response:\n" + response
    else:
        text = prompt + "### Instruction:\n" +  instruction + "\n###Input:\n" + input_query + "\n###Response :\n" + response
        
    text_col.append(text)

alpaca_finance_df.loc[:,"text"] = text_col

alpaca_finance_df.to_csv('./alpaca_finance_df.csv',index=False)

import torch
from datasets import load_dataset,Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer


def train():
    train_dataset = Dataset.from_pandas(alpaca_finance_df)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf", load_in_8bit=True, torch_dtype=torch.float16, device_map="auto"
    )
    model.resize_token_embeddings(len(tokenizer))
    model = prepare_model_for_int8_training(model)
    peft_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_config)

    training_args = TrainingArguments(
        output_dir="llama_v2_finance_alpaca",
        per_device_train_batch_size=4,
        optim="adamw_torch",
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        learning_rate=2e-4,
        fp16=True,
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        num_train_epochs=3
    )
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=1024,
        tokenizer=tokenizer,
        args=training_args,
        packing=True,
        peft_config=peft_config,
    )
    trainer.train()

if __name__ == "__main__":
    with torch.autocast("cuda"): 
        train()
