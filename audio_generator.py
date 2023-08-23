import torch
import scipy
import numpy as np
from transformers import AutoProcessor, AutoModel, AutoModelForCausalLM, BarkModel, LlamaModel


device = torch.device('cuda')
processor = AutoProcessor.from_pretrained("suno/bark-small", device=device,)
model = AutoModel.from_pretrained("suno/bark-small").to(device)
out = np.load('out.npy')
output = torch.from_numpy(out).unsqueeze(0).cuda()
audio = model.codec_decode(output)

sampling_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sampling_rate, data=audio.detach().cpu().numpy().squeeze())
