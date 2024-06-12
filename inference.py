import os
import torch
from huggingface_hub import login
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    MarianMTModel, 
    MarianTokenizer
)
from peft import (
    PeftModel,
    get_peft_config,
    get_peft_model,
    LoraConfig,
    load_peft_weights,
    set_peft_model_state_dict
)
from trl import SFTTrainer
import pandas as pd
import gradio as gr


huggingface_token = "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"
login(token=huggingface_token)
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

compute_dtype = getattr(torch, "float16")
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quant_config,
    device_map="auto",
)
model.config.use_cache = False
model.config.pretraining_tp = 1

peft_model = PeftModel.from_pretrained(model, "./results/checkpoint-2000")
model = peft_model.merge_and_unload()

en_to_uk_model_name = "Helsinki-NLP/opus-mt-en-uk"
en_to_uk_tokenizer = MarianTokenizer.from_pretrained(en_to_uk_model_name)
en_to_uk_model = MarianMTModel.from_pretrained(en_to_uk_model_name)
modelt_name = "Helsinki-NLP/opus-mt-uk-en"
tokenizer_t = MarianTokenizer.from_pretrained(modelt_name)
model_t = MarianMTModel.from_pretrained(modelt_name)

def greet(prompt, lang, temperature, tags):
    if lang == "ukrainian":
        translated = model_t.generate(**tokenizer_t(prompt, return_tensors="pt", padding=True))
        translated_text = [tokenizer_t.decode(t, skip_special_tokens=True) for t in translated]
        prompt = translated_text[0]

    prompt = [{"role": "You are the best instagram post generator in the world!",
           "content": str(prompt + " Write only instagram post, without all explanation information.")}]
    input_ids = tokenizer.apply_chat_template(
        prompt,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        padding=False
    )
    terminators = [
      tokenizer.eos_token_id,
      tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    output = tokenizer.decode(response, skip_special_tokens=True)

    if lang == "ukrainian":
        translated = en_to_uk_model.generate(**en_to_uk_tokenizer(output, return_tensors="pt", padding=True))
        translated_text = [en_to_uk_tokenizer.decode(t, skip_special_tokens=True) for t in translated]
        output = translated_text[0]
    return output  + " " + tags

demo = gr.Interface(
    fn=greet,
    inputs=[gr.Textbox(info="describe the Instagram post"),
            gr.Radio(["english", "ukrainian"], value="english"),
            gr.Slider(minimum=0, maximum=2, value=0.7),
            gr.Textbox(placeholder="#YourHashtag")],
    outputs=["text"],
)

demo.launch()