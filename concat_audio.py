import os
import numpy as np
import soundfile as sf

SILENCE = np.zeros((24000, ))
directory = "/home/ubuntu/test_small/"

sub_dirs = ["either_1", "either_2", "either_3", "short_1", "short_2"]
for SUB_DIRECTORY in sub_dirs:
    files = os.listdir(f"{directory}/{SUB_DIRECTORY}")
    audios = []
    for i in range(100):
        wav, sr = sf.read(f"{directory}/{SUB_DIRECTORY}/audio_{i}.mp3")
        audios.append(wav)
        audios.append(SILENCE)
    total_audio = np.concatenate(audios)
    sf.write(f'/home/ubuntu/{SUB_DIRECTORY}.mp3', total_audio, samplerate=24000)
