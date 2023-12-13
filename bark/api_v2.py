import os
from typing import Dict, Optional, Union
import torchaudio
import soundfile as sf
import numpy as np
import time
from audiostretchy.stretch import AudioStretch
from .generation_v2 import codec_decode, generate_coarse, generate_fine, generate_text_semantic
import audioop
import torch
from vocos import Vocos
from .generation_v2 import generate_coarse, generate_fine, generate_text_semantic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocos = Vocos.from_pretrained("charactr/vocos-encodec-24khz").to(device)


def numpy_audioop_helper(x, xdtype, func, width, ydtype):
    '''helper function for using audioop buffer conversion in numpy'''
    xi = np.asanyarray(x).astype(xdtype)
    if np.any(x != xi):
        xinfo = np.iinfo(xdtype)
        raise ValueError("input must be %s [%d..%d]" % (xdtype, xinfo.min, xinfo.max))
    y = np.frombuffer(func(xi.tobytes(), width), dtype=ydtype)
    return y.reshape(xi.shape)


def audioop_ulaw_compress(x):
    return numpy_audioop_helper(x, np.int16, audioop.lin2ulaw, 2, np.uint8)


def audioop_ulaw_expand(x):
    return numpy_audioop_helper(x, np.uint8, audioop.ulaw2lin, 2, np.int16)

def get_zero_value_length(wav):
    non_zero_indices = np.nonzero(wav)[0]
    if len(non_zero_indices):
        last_non_zero_index = non_zero_indices[-1]
        return last_non_zero_index, len(wav) - last_non_zero_index - 1
    else:
        last_non_zero_index = len(wav) - 1
        return last_non_zero_index, 0


def stretch_wav(wav, rate):
    audio_stretch = AudioStretch()
    audio_stretch.nchannels = 1
    audio_stretch.sampwidth = 2
    audio_stretch.framerate = 8000
    audio_stretch.nframes = len(wav)
    audio_stretch.in_samples = wav

    audio_stretch.stretch(
        ratio=rate,
        gap_ratio=0.0,
        upper_freq=333,
        lower_freq=55,
        buffer_ms=25,
        threshold_gap_db=-40,
        double_range=False,
        fast_detection=False,
        normal_detection=False,
    )
    target_samples = audio_stretch.samples
    _, initial_silence_length = get_zero_value_length(wav)
    target_silence_length = int(initial_silence_length * rate)
    last_non_zero_index, _ = get_zero_value_length(target_samples)
    return audio_stretch.samples[:last_non_zero_index + target_silence_length]

def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    min_eos_p: float = 0.2
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True,
        min_eos_p=min_eos_p
    )
    return x_semantic


def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
    stream=None,
    initial_index=0,
    min_eos_p=0.2,
    rate=1.0
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    print(text)
    last_audio = None
    index = initial_index
    x_coarse_in = None
    n_step = 0
    cnt = 0
    def gen_audio_from_coarse(last_audio, index, is_last=False):
        fine_tokens = generate_fine(
            coarse_tokens,
            history_prompt=history_prompt,
            temp=0.5,
        )
        # fine_tokens = coarse_tokens
        audio_tokens_torch = torch.from_numpy(fine_tokens).to(device)
        features = vocos.codes_to_features(audio_tokens_torch)
        audio_arr = torchaudio.functional.resample(
            vocos.decode(features, bandwidth_id=torch.tensor([2], device=device)).squeeze(),
            orig_freq=24000,
            new_freq=8000
        ).cpu().numpy()
        if last_audio is None:
            start = 0
            end_point = len(audio_arr) - int(0.2 * 8000)
            last_audio = audio_arr[:end_point]
        else:
            start = len(last_audio)
            audio_arr[:len(last_audio)] = last_audio
            end_point = len(audio_arr) - int(0.2 * 8000) if not is_last else len(audio_arr)
            last_audio = audio_arr[:end_point]
        audio_mu = audioop_ulaw_compress(stretch_wav(np.int16(audio_arr[start:end_point] * 2**15), rate))
        if stream is not None:
            stream.put(audio_mu.tobytes())
        print(f"audio_{index}.raw", time.time())
        index += 1
        return last_audio, index
    for semantic_tokens, is_finished in text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
        min_eos_p=min_eos_p,
    ):
        coarse_tokens, x_coarse_in, n_step = generate_coarse(
            semantic_tokens,
            is_finished,
            history_prompt=history_prompt,
            temp=waveform_temp,
            silent=silent,
            use_kv_caching=True,
            initial_x_coarse_in=x_coarse_in,
            initial_n_step=n_step
        )
        last_audio, index = gen_audio_from_coarse(last_audio, index, is_last=is_finished)
    # last_audio, index = gen_audio_from_coarse(last_audio, index, is_last=True)
        
    # print("Total Audio Length: ", len(audio_arr) / 24000)
    return index

