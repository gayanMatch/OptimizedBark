import time
import asyncio
import queue
import os
from threading import Thread
from bark.synthesize import synthesize

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
        self.synthesize_queue = queue.Queue()
        self.isWorking = True
        self.voice = voice
        self.directory = "bark/static"
        self.request_dict = dict()
        self.request_num = 0

    def add_request(self, text, voice, rate=1.0):
        request_id = f"CA{self.request_num}"
        self.request_num += 1
        stream = AsyncStream(request_id)
        self.synthesize_queue.put_nowait((stream, {"text": text, "voice": voice, "rate": rate}))
        return stream

    def run(self) -> None:
        synthesize("Hello, this is warm up synthesize.")
        self.isWorking = False
        while True:
            stream, kwargs = self.synthesize_queue.get()
            print("Synthesis Started: ", time.time())
            self.isWorking = True
            synthesize(kwargs["text"], stream, voice=kwargs["voice"], rate=kwargs["rate"])
            print("Synthesize Finished:", time.time())
            print("Synthesize Finished:", kwargs["text"])
            self.isWorking = False

    def is_busy(self) -> bool:
        return not self.synthesize_queue.empty() or self.isWorking