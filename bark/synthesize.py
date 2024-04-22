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
    file = None
    if stream is None:
        file = open("synthesized.wav", "wb")
        header_16000_file = open('bark/assets/header_16000.raw', 'rb')
        header_16000 = header_16000_file.read()
        header_16000_file.close()
        file.write(header_16000)
    for sentence in syn_sentences:
        if sentence:
            if word_count(sentence) < 5:
                index = generate_audio(
                    sentence,
                    history_prompt=voice,
                    text_temp=0.7,
                    waveform_temp=0.5,
                    silent=True,
                    stream=stream,
                    file=file,
                    initial_index=index,
                    rate=rate,
                    min_eos_p=0.1
                )
            else:
                index = generate_audio(
                    sentence,
                    history_prompt=voice,
                    text_temp=0.7,
                    waveform_temp=0.5,
                    silent=True,
                    stream=stream,
                    file=file,
                    initial_index=index,
                    rate=rate
                )
    if stream is not None:
        stream.finish()
    elif file is not None:
        file.close()
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


if __name__ == "__main__":
    print("Synthesize Ready")
    clip = "Hi, this is warm up synthesize."
    while True:
        synthesize(clip)
        clip = input("Type your text here: \n")
