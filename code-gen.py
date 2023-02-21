import torch
from transformers import AutoTokenizer, CodeGenForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("Salesforce/codegen-2B-mono")
model = CodeGenForCausalLM.from_pretrained("Salesforce/codegen-2B-mono")
model.to('cuda')
inputs = tokenizer("write a function to add 2 numbers", return_tensors="pt")
input_ids = inputs["input_ids"].to('cuda')
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

gen_config = GenerationConfig(
    do_sample=False,
    max_new_tokens=500,
    use_cache=False,
    pad_token_id=tokenizer.pad_token_id,
)


output_bt = model.generate(input_ids, generation_config=gen_config)
print(tokenizer.decode(output_bt[0], skip_special_tokens=True))

