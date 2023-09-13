import time
import os
import soundfile as sf
import pickle
import nltk
from bark.api import generate_audio, save_as_prompt
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic



def synthesize(text_prompt, directory="static", voice="en_fiery", index=0):
    start_time = time.time()
    text = text_prompt.replace("\n", " ").strip()
    # sentences = nltk.sent_tokenize(text)
    
    # for sentence in sentences:
    #     index = generate_audio(sentence, history_prompt=voice.replace('.npz', ''), directory=directory, initial_index=index, silent=True)
    prompt, audio = generate_audio(text, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
    # save_as_prompt(f'{directory}/prompt.npz', prompt)
    # sf.write(f"{directory}/audio_{index}.mp3", audio, samplerate=SAMPLE_RATE)
    # file = open(f'{directory}/finish.lock', 'wt')
    # file.write("Finish")
    # file.close()
    # generate_audio(text_prompt, history_prompt="en_fiery", directory=directory, initial_index=index, silent=True)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


if __name__ == "__main__":
    preload_models(text_use_small=True)
    print("Synthesize Ready")
    text_prompt = """
It looks like you opted into one of our ads lookin' for information on how to scale your business using AI. Do you remember that?
Hello, I'm really excited about optimizing bark with Air AI.
"""
    test_clip = "Hello, Thanks for visiting our bebe company. My name is Mark Fiery and I'm the sales assistant. How can I help you?"
    clip = "Hi, this is warm up synthesize."
    import sys
    if sys.platform.startswith('win'):
        directory = 'static'
    else:
        directory = 'bark/static'
    audio_array = synthesize(clip, directory=directory)

    file = open('/home/ubuntu/test_small.txt')
    lines = file.readlines()
    os.makedirs("/home/ubuntu/test_small/either_1", exist_ok=True)
    os.makedirs("/home/ubuntu/test_small/either_2", exist_ok=True)
    os.makedirs("/home/ubuntu/test_small/either_3", exist_ok=True)
    os.makedirs("/home/ubuntu/test_small/short_1", exist_ok=True)
    os.makedirs("/home/ubuntu/test_small/short_2", exist_ok=True)
    time_array = []
    for i in range(30):
        time_array.append([])
    
    for i, line in enumerate(lines[:100]):
        # text = "Totally get it, brother. Okay, no worries. Okay, well, with the video, you did get a chance to watch it though, right?"
        # audio_array = synthesize(text, directory=directory, voice="bark/static/prompt.npz")
        voice = "final_Either_way_weve-23_09_04__17-51-24.mp4"
        if len(line.split()) < 5:
            voice = "short"
        word_length = len(line.split())
        s = time.time()
        synthesize(line, voice=voice, index=i, directory="/home/ubuntu/test_small/either_1")
        time_array[word_length].append(time.time() - s)
        s = time.time()
        synthesize(line, voice=voice, index=i, directory="/home/ubuntu/test_small/either_2")
        time_array[word_length].append(time.time() - s)
        s = time.time()
        synthesize(line, voice=voice, index=i, directory="/home/ubuntu/test_small/either_3")
        time_array[word_length].append(time.time() - s)
        # synthesize(line, voice="Hey_Justin_How_-23_09_06__05-48-30.mp3_original_speaker_1", index=i, directory="/home/ubuntu/test_small/short_1")
        # synthesize(line, voice="Hey_Justin_How_-23_09_06__05-48-30.mp3_original_speaker_1", index=i, directory="/home/ubuntu/test_small/short_2")
    pickle.dump(time_array, open('time_array.pkl', 'wb'))