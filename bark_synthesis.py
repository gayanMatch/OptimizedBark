import os

import nltk

from bark.synthesize import synthesize

if __name__ == "__main__":
    nltk.download('punkt')
    print("Synthesize Ready")
    clip = "Hi, this is warm up synthesize."
    import sys

    if sys.platform.startswith('win'):
        directory = 'static'
    else:
        directory = 'bark/static'
    os.makedirs(directory, exist_ok=True)
    audio_array = synthesize(clip, voice="hey_james_reliable_1_small_coarse_fix")
    text = ("With those in mind, let's break it down. Our conversational AI has a proven track record of improving "
            "lead conversion by 25-35%. That means you could potentially see a CLV increase to about $12,500.")
    audio_array = synthesize(text, voice="final_Either_way_weve-23_09_04__17-51-24.mp4")
