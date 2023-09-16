import queue
import time
import torch
import soundfile as sf
import numpy as np
from threading import Thread
from bark.generation_v3 import load_model
from bark.generation_v3 import generate_text_semantic, generate_coarse, generate_fine
from vocos import Vocos

def detect_last_silence_index(audio_data, sr=24000, threshold=0.0015, min_silence=25):
    silence_start = None
    silence_count = 0
    min_silent_samples = (sr * min_silence) // 1000

    for i in range(len(audio_data) - 1, -1, -1):
        if abs(audio_data[i]) <= threshold:
            if silence_start is None:
                silence_start = i
            silence_count += 1
        else:
            if silence_start is not None and silence_count >= min_silent_samples:
                break
            silence_start = None
            silence_count = 0

    return silence_start - silence_count // 2 if silence_start is not None else None

class BarkThread(Thread):
    def __init__(self, queue):
        super().__init__()
        self.text_queue = queue
    
    def run(self):
        pass

class SemanticThread(Thread):
    def __init__(self, text_queue, semantic_queue, voice="final_Either_way_weve-23_09_04__17-51-24.mp4"):
        super().__init__()
        self.in_queue = text_queue
        self.model = load_model(model_type="text", use_gpu=True, use_small=True, force_reload=False)
        self.out_queue = semantic_queue
        self.voice = voice

    def run(self):
        while True:
            text = self.in_queue.get()
            generate_text_semantic(
                text,
                history_prompt=self.voice,
                use_kv_caching=True,
                model_container=self.model,
                queue=self.out_queue
            )

class CoarseThread(Thread):
    def __init__(self, semantic_queue, coarse_queue, voice="final_Either_way_weve-23_09_04__17-51-24.mp4"):
        super().__init__()
        self.in_queue = semantic_queue
        self.out_queue = coarse_queue
        self.model = load_model(
            model_type="coarse",
            use_gpu=True,
            use_small=True,
            force_reload=False,
        )
        self.voice = voice

    def run(self):
        while True:
            _ = self.in_queue.get()
            generate_coarse(
                self.in_queue,
                self.out_queue,
                history_prompt=self.voice,
                use_kv_caching=True,
                silent=True,
                temp=0.5,
                model_container=self.model
            )

class AudioThread(Thread):
    def __init__(self, coarse_queue, directory, voice="final_Either_way_weve-23_09_04__17-51-24.mp4"):
        super().__init__()
        self.in_queue = coarse_queue
        self.fine_model = load_model(
            model_type="fine",
            use_gpu=True,
            use_small=False,
            force_reload=False
        )
        self.vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").cuda()
        self.last_audio = None
        self.directory = directory
        self.index = 0
        self.is_working = False
        self.voice = voice

    def run(self):
        while True:
            coarse_tokens = self.in_queue.get()
            self.is_working = True
            fine_tokens = generate_fine(
                coarse_tokens,
                history_prompt=self.voice,
                temp=0.5,
            )
            audio_tokens_torch = torch.from_numpy(fine_tokens).to('cuda')
            features = self.vocos.codes_to_features(audio_tokens_torch)
            audio_arr = self.vocos.decode(features, bandwidth_id=torch.tensor([2], device='cuda')).cpu().numpy()[0]
            if self.last_audio is None:
                self.last_audio = audio_arr
                start = 0
                end_point = len(audio_arr)
            else:
                start = len(self.last_audio)
                audio_arr[:len(self.last_audio)] = self.last_audio
                end_point = detect_last_silence_index(audio_arr)
                self.last_audio = audio_arr[:end_point]
            sf.write(f"{self.directory}/audio_{self.index}.mp3", np.float32(audio_arr[start:end_point]), 24000)
            self.is_working = False or not self.in_queue.empty()

if __name__ == "__main__":
    text_queue = queue.Queue()
    semantic_queue = queue.Queue()
    semantic_thread = SemanticThread(text_queue, semantic_queue)
    semantic_thread.start()
    text_queue.put("Hi, this is speech synthesis test.")
    while True:
        time.sleep(0.5)
        print(semantic_queue.get())
