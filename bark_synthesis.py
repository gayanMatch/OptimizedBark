import os
import nltk
import time
import shutil
from bark.synthesize import synthesize


if __name__ == "__main__":
    nltk.download('punkt')
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
    # while True:
    #     synthesize(clip, directory=directory)
    #     # synthesize(test_clip, directory=directory)
    #     clip = input("Type your text here: \n")
    os.makedirs(directory, exist_ok=True)
    audio_array = synthesize(clip, directory=directory)
    text = "With those in mind, let's break it down. Our conversational AI has a proven track record of improving lead conversion by 25-35%. That means you could potentially see a CLV increase to about $12,500."
    # audio_array = synthesize(text, directory=directory, voice="bark/static/prompt.npz")
    audio_array = synthesize(text, directory=directory, voice="final_Either_way_weve-23_09_04__17-51-24.mp4")

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=1)

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=2)
    # print(audio_array.shape)
    # sf.write('bark/static/audio.mp3', audio_array, samplerate=24000)