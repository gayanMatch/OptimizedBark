import os
import time
import soundfile as sf
import nltk
from bark.api_v2 import generate_audio, save_as_prompt, generate_prompt


def word_count(sentence):
    return len(sentence.split(' '))


def synthesize(text_prompt, directory="static", voice="en_fiery"):
    start_time = time.time()
    text = text_prompt.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text)
    index = 0
    last_sentence = ''
    syn_sentences = []
    for sentence in sentences:
        if word_count(last_sentence + ' ' + sentence) > 30:
            syn_sentences.append(last_sentence)
            last_sentence = sentence
        else:
            last_sentence = last_sentence + (' ' if last_sentence else '') + sentence
    if last_sentence:
        syn_sentences.append(last_sentence)

    for sentence in syn_sentences:
        if word_count(sentence) < 7:
            if not os.path.exists(f"bark/assets/prompts/short/{voice}.npz"):
                synthesize_prompt(voice)
            index = generate_audio(sentence, history_prompt=f"short/{voice}", text_temp=0.7, waveform_temp=0.5, silent=True, directory=directory, initial_index=index)
        else:
            index = generate_audio(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, directory=directory, initial_index=index)
    file = open(f'{directory}/finish.lock', 'wt')
    file.write("Finish")
    file.close()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


def synthesize_prompt(voice):
    text_prompt = "I'm going to speak short prompts. Only short prompt itself."
    generate_prompt(text_prompt, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=False)
    

if __name__ == "__main__":
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
