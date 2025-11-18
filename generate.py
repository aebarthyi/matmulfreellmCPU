import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.profiler import profile, record_function, ProfilerActivity
#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, device_map='cpu').half()
input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    with record_function("model_inference"):
        outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("cpu_trace.json")