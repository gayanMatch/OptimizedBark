import time
import os
from threading import Thread
from bark.synthesize import synthesize, synthesize_prompt
class SynthesizeThread(Thread):
    def __init__(self, voice):
        super().__init__()
        self.synthesize_queue = []
        self.isWorking = False
        self.voice = voice
        self.directory = "bark/static"
    def run(self) -> None:
        synthesize("Hello, this is warm up synthesize.", directory=self.directory)
        while True:
            if self.synthesize_queue:
                print("Synthesis Started: ", time.time())
                self.isWorking = True

                for args, directory in self.synthesize_queue:
                    print("Starting Synthesis")
                    synthesize(**args, directory=directory)
                    # time.sleep(2)
                    print("Synthesize Finished:", time.time())
                    print("Synthesize Finished:", args["text_prompt"])
                self.synthesize_queue = []
                self.isWorking = False
            
            time.sleep(0.01)
