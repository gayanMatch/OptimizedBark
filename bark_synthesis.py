import time
import nltk
from bark.generation import preload_models
from bark import generate_audio, SAMPLE_RATE

def synthesize(text_prompt, directory="static", index_=0):
    start_time = time.time()
    index = index_
    generate_audio(text_prompt, history_prompt="v2/en_speaker_9", directory=directory, initial_index=index)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Time for syntesize: {duration}")


if __name__ == "__main__":
    preload_models()
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
    synthesize(clip, directory=directory)
    text = "Ok. Perfect. Well the reason I’m calling you is I’m actually a manager over here on Scale 13's client success team on a recorded line,"
    # synthesize(text, directory=directory)

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=1)

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=2)
