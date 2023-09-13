import time
import soundfile as sf
import nltk
from bark.api import generate_audio, save_as_prompt
from transformers import BertTokenizer
from bark.generation import SAMPLE_RATE, preload_models, codec_decode, generate_coarse, generate_fine, generate_text_semantic



def synthesize(text_prompt, directory="static", voice="en_fiery", index_=0):
    start_time = time.time()
    text = text_prompt.replace("\n", " ").strip()
    # sentences = nltk.sent_tokenize(text)
    
    # for sentence in sentences:
    #     index = generate_audio(sentence, history_prompt=voice.replace('.npz', ''), directory=directory, initial_index=index, silent=True)
    prompt, audio = generate_audio(text, history_prompt=voice, text_temp=0.7, waveform_temp=0.5, silent=True, output_full=True)
    save_as_prompt(f'{directory}/prompt.npz', prompt)
    sf.write(f"{directory}/audio_0.mp3", audio, samplerate=SAMPLE_RATE)
    file = open(f'{directory}/finish.lock', 'wt')
    file.write("Finish")
    file.close()
    # generate_audio(text_prompt, history_prompt="en_fiery", directory=directory, initial_index=index, silent=True)
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
    audio_array = synthesize(clip, directory=directory)
    text = "Totally get it, brother. Okay, no worries. Okay, well, with the video, you did get a chance to watch it though, right?"
    # audio_array = synthesize(text, directory=directory, voice="bark/static/prompt.npz")
    audio_array = synthesize(text, directory=directory, voice="final_Either_way_weve-23_09_04__17-51-24.mp4")

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=1)

    # # text = "Yeah. So it uh, it looks like you opted into one of our ads looking for information on how to scale your business using AI."
    # synthesize(text, directory=directory, index_=2)
    # print(audio_array.shape)
    # sf.write('bark/static/audio.mp3', audio_array, samplerate=24000)