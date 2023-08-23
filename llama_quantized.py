import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
if __name__ == '__main__':
   model_id = "meta-llama/Llama-2-7b-chat-hf"
   prompt = "You should answer in one sentence. What is Github?"
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')
   model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=nf4_config)
   # model_nf4 = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()
   streamer = TextIteratorStreamer(tokenizer,
                                   timeout=10.,
                                   skip_prompt=True,
                                   skip_special_tokens=True)
   s = time.time()
   model_nf4.generate(**inputs, streamer=streamer, temperature=0.01)
   output = []
   for text in streamer:
      output.append(text)
   print(''.join(output))
   print(time.time() - s)

