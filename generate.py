import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import mmfreelm
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from torch.profiler import profile, ProfilerActivity, tensorboard_trace_handler
import traceback
from mmfreelm.ops.fusedbitnet import BitLinear

torch.set_num_threads(8)              # parallelism within a single op (matmuls etc.)
torch.set_num_interop_threads(4)

#Change here to our open-sourced model
name = 'ridger/MMfreeLM-370M'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, device_map='cpu').half()

for module in model.modules():
    if isinstance(module, BitLinear) and 'lm_head' not in name:
        module.prepare_for_inference()

input_prompt = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, "
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

with torch.no_grad():
    _ = model.generate(input_ids, max_length=32)

with profile(
    activities=[ProfilerActivity.CPU],
    on_trace_ready=tensorboard_trace_handler("./log/profile"),
    record_shapes=True,
    with_stack=True,
) as prof:
        outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)

print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
prof.export_chrome_trace("cpu_trace.json")

outputs = model.generate(input_ids, pad_token_id=tokenizer.eos_token_id, max_length=32, do_sample=True, top_p=0.4, temperature=0.6)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])