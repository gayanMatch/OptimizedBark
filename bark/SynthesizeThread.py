import time
import asyncio
import queue
import os
from threading import Thread
from bark.synthesize import synthesize, synthesize_prompt

class AsyncStream:
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue = asyncio.Queue()
        self._finished = False

    def put(self, item) -> None:
        if self._finished:
            return
        self._queue.put_nowait(item)

    def finish(self) -> None:
        self._queue.put_nowait(StopIteration)
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __aiter__(self):
        return self

    async def __anext__(self):
        result = await self._queue.get()
        if result is StopIteration:
            raise StopAsyncIteration
        elif isinstance(result, Exception):
            raise result
        return result

class SynthesizeThread(Thread):
    def __init__(self, voice):
        super().__init__()
        self.synthesize_queue = []
        self.isWorking = False
        self.voice = voice
        self.directory = "bark/static"
        self.request_dict = dict()


    def add_request(self, text, voice):
        request_id = ""


    def run(self) -> None:
        synthesize("Hello, this is warm up synthesize.", directory=self.directory)
        while True:
            if self.synthesize_queue:
                print("Synthesis Started: ", time.time())
                self.isWorking = True

                for sentence, directory in self.synthesize_queue:
                    print("Starting Synthesis")
                    synthesize(sentence, directory=directory, voice=self.voice)
                    # time.sleep(2)
                    print("Synthesize Finished:", time.time())
                    print("Synthesize Finished:", sentence)
                self.synthesize_queue = []
                self.isWorking = False
            
            time.sleep(0.01)
