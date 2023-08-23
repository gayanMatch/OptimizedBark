import torch
import scipy
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, BarkModel, LlamaModel
from transformers import BitsAndBytesConfig

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)
# AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")


device = torch.device('cuda')
processor = AutoProcessor.from_pretrained("suno/bark-small", device=device,)
model = AutoModel.from_pretrained("suno/bark-small", torch_dtype=torch.float16).to_bettertransformer().to(device)
# model = AutoModel.from_pretrained("suno/bark-small", quantization_config=nf4_config)
# print("THHHHHH")
# model = AutoModel.from_pretrained("suno/bark-small").to(device)


voices = {
    "male": "v2/en_speaker_6",
    "female": "v2/en_speaker_9"
}

text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
import time
s = time.time()
inputs = processor(text=text, return_tensors="pt", voice_preset=voices['female'])
speech, output = model.generate(**inputs.to(device))
print(time.time() - s)

s = time.time()
inputs = processor(text=text, return_tensors="pt", voice_preset=voices['female'])
speech, output = model.generate(**inputs.to(device))
print(time.time() - s)
#
# s = time.time()
# inputs = processor(text=text, return_tensors="pt", voice_preset=voices['female'])
# speech = model.generate(**inputs.to(device))
# print(time.time() - s)

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=speech.cpu().numpy().squeeze())
