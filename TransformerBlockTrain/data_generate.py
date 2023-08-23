import yaml
import re
import os
import sys
import time
import random
import logging
import funcy
import numpy as np
import torch
import contextlib
import torch.nn.functional as F
import tqdm
from transformers import BertTokenizer
from model import GPTConfig, GPT

SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000
SEMANTIC_PAD_TOKEN = 10_000
SEMANTIC_INFER_TOKEN = 129_599
TEXT_ENCODING_OFFSET = 10_048
TEXT_PAD_TOKEN = 129_595

TRAIN_DIR = '/datasets/val'
CUR_PATH = "."
SUPPORTED_LANGS = [
    ("English", "en"),
]
ALLOWED_PROMPTS = ["announcer", "en_fiery"]
for _, lang in SUPPORTED_LANGS:
    for prefix in ("", f"v2{os.path.sep}"):
        for n in range(10):
            ALLOWED_PROMPTS.append(f"{prefix}{lang}_speaker_{n}")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
gpt_config = GPTConfig(**yaml.safe_load(open('gpt_config.yaml')))
model = GPT(gpt_config)
state_dict = torch.load('semantic.pth')
model.load_state_dict(state_dict)
model.eval()
model.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
logger = logging.getLogger(__name__)

if (
    torch.cuda.is_available() and
    hasattr(torch.cuda, "amp") and
    hasattr(torch.cuda.amp, "autocast") and
    hasattr(torch.cuda, "is_bf16_supported") and
    torch.cuda.is_bf16_supported() and
    False
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.bfloat16)
else:
    @contextlib.contextmanager
    def autocast():
        yield


class InferenceContext:
    def __init__(self, benchmark=False):
        # we can't expect inputs to be the same length, so disable benchmarking by default
        self._chosen_cudnn_benchmark = benchmark
        self._cudnn_benchmark = None

    def __enter__(self):
        self._cudnn_benchmark = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = self._chosen_cudnn_benchmark

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self._cudnn_benchmark


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


@contextlib.contextmanager
def _inference_mode():
    with InferenceContext(), torch.inference_mode(), torch.no_grad(), autocast():
        yield


def _clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


def _load_history_prompt(history_prompt_input):
    if isinstance(history_prompt_input, str) and history_prompt_input.endswith(".npz"):
        history_prompt = np.load(history_prompt_input)
    elif isinstance(history_prompt_input, str):
        # make sure this works on non-ubuntu
        history_prompt_input = os.path.join(*history_prompt_input.split("/"))
        if history_prompt_input not in ALLOWED_PROMPTS:
            raise ValueError("history prompt not found")
        history_prompt = np.load(
            os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt_input}.npz")
        )
    elif isinstance(history_prompt_input, dict):
        assert ("semantic_prompt" in history_prompt_input)
        assert ("coarse_prompt" in history_prompt_input)
        assert ("fine_prompt" in history_prompt_input)
        history_prompt = history_prompt_input
    else:
        raise ValueError("history prompt format unrecognized")
    return history_prompt


def generate_text_semantic(
        text,
        history_prompt=None,
        temp=0.7,
        silent=False,
        min_eos_p=0.2,
        max_gen_duration_s=None,
        allow_early_stop=True,
        use_kv_caching=False,
        index=0,
):
    """Generate semantic tokens from text."""
    assert isinstance(text, str)
    text = _normalize_whitespace(text)
    assert len(text.strip()) > 0
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        semantic_history = history_prompt["semantic_prompt"]
        assert (
                isinstance(semantic_history, np.ndarray)
                and len(semantic_history.shape) == 1
                and len(semantic_history) > 0
                and semantic_history.min() >= 0
                and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
        )
    else:
        semantic_history = None
    # load models if not yet exist
    global model, tokenizer
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([
            encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
        ]).astype(np.int64)
    )[None]
    # assert x.shape[1] == 256 + 256 + 1
    with _inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=n_tot_steps)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache_new = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            data = {'x_input': x_input.cpu().numpy(), 'logits': logits.cpu().numpy()}
            np.savez(f"{TRAIN_DIR}/{voice.replace('/', '_')}_{index}_{x_input.cpu().numpy().shape[1]}.npz", **data)
            kv_cache = kv_cache_new
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            probs = F.softmax(relevant_logits / temp, dim=-1)
            # multinomial bugged on mps: shuttle to cpu if necessary
            inf_device = probs.device
            if probs.device.type == "mps":
                probs = probs.to("cpu")
            item_next = torch.multinomial(probs, num_samples=1)
            probs = probs.to(inf_device)
            item_next = item_next.to(inf_device)
            if allow_early_stop and (
                    item_next == SEMANTIC_VOCAB_SIZE
                    or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(n - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(n - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(n - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            if n > pbar_state:
                if n > pbar.total:
                    pbar.total = n
                pbar.update(n - pbar_state)
            pbar_state = n
        pbar.total = n
        pbar.refresh()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1:]
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()
    return out, index


if __name__ == '__main__':
    os.makedirs('/datasets', exist_ok=True)
    DIR_dict = {
        'val': '/datasets/val',
        'train': '/datasets/train',
    }
    file_dict = {
        'val': 'val.txt',
        'train': 'train.txt'
    }
    generate_num = {
        'val': 50,
        'train': 500
    }

    mode = sys.argv[1]
    TRAIN_DIR = DIR_dict[mode]
    os.makedirs(TRAIN_DIR, exist_ok=True)
    file = open(file_dict[mode])
    data = file.readlines()[:generate_num[mode]]
    file.close()
    train_data_index = 0
    for i, line in enumerate(data):
        s = time.time()
        line = line.strip()
        # voice = random.choice(ALLOWED_PROMPTS)
        for voice in ALLOWED_PROMPTS:
            _, index = generate_text_semantic(line, history_prompt=voice, temp=0.7, silent=False, use_kv_caching=False,
                                              index=train_data_index)
            train_data_index += 1
        print(time.time() - s, i)
