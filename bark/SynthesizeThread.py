import time
from threading import Thread
from bark.synthesize import synthesize
class SynthesizeThread(Thread):
    def __init__(self, voice):
        super().__init__()
        self.synthesize_queue = []
        self.isWorking = False
        self.voice = voice
    def run(self) -> None:
        synthesize("Hello, this is warm up synthesize.", directory="bark/static")
        while True:
            if self.synthesize_queue:
                print("Synthesis Started: ", time.time())
                self.isWorking = True
                for sentence in self.synthesize_queue:
                    synthesize(sentence, directory="bark/static", voice=self.voice)
                    # time.sleep(2)
                    print("Synthesize Finished:", time.time())
                    print("Synthesize Finished:", sentence)
                self.synthesize_queue = []
                self.isWorking = False
            time.sleep(0.01)
