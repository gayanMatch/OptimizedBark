import time
import nltk
from bark.api_v2 import generate_audio


def word_count(sentence):
    return len(sentence.split(' '))


def synthesize(text="", stream=None, voice="en_fiery", rate=1.0):
    start_time = time.time()
    text_prompt = text.replace("\n", " ").strip()
    sentences = nltk.sent_tokenize(text_prompt)
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
        if sentence:
            if word_count(sentence) < 5:
                index = generate_audio(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, stream=stream, initial_index=index, rate=rate, min_eos_p=0.1)
            else:
                index = generate_audio(sentence, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, stream=stream, initial_index=index, rate=rate)
    if stream is not None:
        stream.finish()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


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
