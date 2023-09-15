import os
import time
import soundfile as sf
import nltk
from bark.api import generate_audio, save_as_prompt, generate_audio_stream
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic


def word_count(sentence):
    return len(sentence.split(' '))


def synthesize(text_prompt, directory="static", voice="en_fiery", index_=0):
    start_time = time.time()
    text = text_prompt.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)
    sentence = sentences[0]
    index = 0
    if word_count(sentence) < 7:
        if not os.path.exists(f"bark/assets/prompts/short/{voice}.npz"):
            synthesize_prompt(voice)
        prompt, audio = generate_audio(sentence, history_prompt=f"short/{voice}", text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
    else:
        prompt, audio = generate_audio(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
    sf.write(f"{directory}/audio_{index}.mp3", audio, samplerate=SAMPLE_RATE)
    print(f"{directory}/audio_{index}.mp3", time.time())
    save_as_prompt(f"{directory}/prompt_{index}.npz", prompt)
    syn_sentences = []
    index += 1
    for i, sentence in enumerate(sentences[1:]):
        if word_count(sentence) < 7:
            if 2 + i < len(sentences):
                sentences[2 + i] = sentences[i + 1] + " " + sentences[i + 2]
            else:
                sentences[i] = sentences[i] + sentences[i + 1]
            
    for sentence in sentences[1:]:
        if word_count(sentence) > 6:
            syn_sentences.append(sentence)

    for sentence in syn_sentences:
        if word_count(sentence) < 7:
            if not os.path.exists(f"bark/assets/prompts/short/{voice}.npz"):
                synthesize_prompt(voice)
            prompt, audio = generate_audio(sentence, history_prompt=f"short/{voice}", text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
            sf.write(f"{directory}/audio_{index}.mp3", audio, samplerate=SAMPLE_RATE)
            save_as_prompt(f"{directory}/prompt_{index}.npz", prompt)
            print(f"{directory}/audio_{index}.mp3")
            index += 1
        elif word_count(sentence) < 15:
            prompt, audio = generate_audio(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
            # _, audio = generate_audio(sentence, history_prompt="final_Either_way_weve-23_09_04__17-51-24.mp4", text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
            sf.write(f"{directory}/audio_{index}.mp3", audio, samplerate=SAMPLE_RATE)
            save_as_prompt(f"{directory}/prompt_{index}.npz", prompt)
            print(f"{directory}/audio_{index}.mp3")
            index += 1
        else:
            index = generate_audio_stream(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, directory=directory, initial_index=index)
    # prompt, audio = generate_audio(text, history_prompt=voice.replace('.npz', ''), text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
    # save_as_prompt(f'{directory}/prompt.npz', prompt)
    # sf.write(f"{directory}/audio_0.mp3", audio, samplerate=SAMPLE_RATE)
    file = open(f'{directory}/finish.lock', 'wt')
    file.write("Finish")
    file.close()
    # generate_audio(text_prompt, history_prompt="en_fiery", directory=directory, initial_index=index, silent=True)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


def synthesize_prompt(voice):
    text_prompt = "I'm going to speak short prompts. Only short prompt itself."
    prompt, _ = generate_audio(text_prompt, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, output_full=True, silent=False)
    save_as_prompt(f"bark/assets/prompts/short/{voice}.npz", prompt)
    

if __name__ == "__main__":
    preload_models()
    print("Synthesize Ready")
    text_prompt = """
It looks like you opted into one of our ads lookin' for information on how to scale your business using AI. Do you remember that?
Hello, I'm really excited about optimizing bark with Air AI.
"""
    test_clip = "Hello, Thanks for visiting our bebe company. My name is Mark Fiery and I'm the sales assistant. How can I help you?"
    clip = "Hi, this is warm up synthesize."
    while True:
        synthesize(clip)
        clip = input("Type your text here: \n")
