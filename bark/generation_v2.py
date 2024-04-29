import contextlib
import gc
import logging
import os
import re

import funcy
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import yaml
from encodec import EncodecModel
from huggingface_hub import hf_hub_download
from scipy.special import softmax
from transformers import BertTokenizer
from vocos import Vocos
from GPT2.trt import GPT2TRTDecoder
from NNDF.models import TRTEngineFile
from .model import GPTConfig, GPT, GPT_COARSE
from .model_fine import FineGPT, FineGPTConfig

if (
        torch.cuda.is_available() and
        hasattr(torch.cuda, "amp") and
        hasattr(torch.cuda.amp, "autocast") and
        hasattr(torch.cuda, "is_bf16_supported") and
        torch.cuda.is_bf16_supported()
):
    autocast = funcy.partial(torch.cuda.amp.autocast, dtype=torch.float16)
else:
    @contextlib.contextmanager
    def autocast():
        yield

# hold models in global scope to lazy load
global models
models = {}

global models_devices
models_devices = {}

CONTEXT_WINDOW_SIZE = 1024
CHUNK_SIZE = 2  # number of chunks for streaming smaller values gives faster response time
SEMANTIC_RATE_HZ = 49.9
SEMANTIC_VOCAB_SIZE = 10_000

CODEBOOK_SIZE = 1024
N_COARSE_CODEBOOKS = 2
N_FINE_CODEBOOKS = 8
COARSE_RATE_HZ = 75

SAMPLE_RATE = 24_000

SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

logger = logging.getLogger(__name__)

CUR_PATH = os.path.dirname(os.path.abspath(__file__))

default_cache_dir = os.path.join(os.path.expanduser("~"), ".cache")
CACHE_DIR = os.path.join(os.getenv("XDG_CACHE_HOME", default_cache_dir), "suno", "bark_v0")


def _cast_bool_env_var(s):
    return s.lower() in ('true', '1', 't')


USE_SMALL_MODELS = _cast_bool_env_var(os.environ.get("SUNO_USE_SMALL_MODELS", "False"))
GLOBAL_ENABLE_MPS = _cast_bool_env_var(os.environ.get("SUNO_ENABLE_MPS", "False"))
OFFLOAD_CPU = _cast_bool_env_var(os.environ.get("SUNO_OFFLOAD_CPU", "False"))

REMOTE_MODEL_PATHS = {
    "text_small": {
        "repo_id": "suno/bark",
        "file_name": "text.pt",
    },
    "coarse_small": {
        "repo_id": "suno/bark",
        "file_name": "coarse.pt",
    },
    "fine_small": {
        "repo_id": "suno/bark",
        "file_name": "fine.pt",
    },
    "text": {
        "repo_id": "suno/bark",
        "file_name": "text_2.pt",
    },
    "coarse": {
        "repo_id": "suno/bark",
        "file_name": "coarse_2.pt",
    },
    "fine": {
        "repo_id": "suno/bark",
        "file_name": "fine_2.pt",
    },
}
LOCAL_MODEL_PATHS = {  # model paths in project repo
    "text_small": "./models/bark/pytorch/model.pt",
    "coarse_small": "./models/bark_coarse/pytorch/model.pt",
    "fine_small": "./models/bark_fine/pytorch/model.pt",
    "text": "./models/bark_large/pytorch/model.pt",
    "coarse": "./models/bark_coarse_large/pytorch/model.pt",
    "fine": "./models/bark_fine_large/pytorch/model.pt",
}

if not hasattr(torch.nn.functional, 'scaled_dot_product_attention') and torch.cuda.is_available():
    logger.warning(
        "torch version does not support flash attention. You will get faster" +
        " inference speed by upgrade torch to newest nightly version."
    )


def _grab_best_device(use_gpu=True):
    """
    Grabs the best available GPU device
    :param use_gpu: whether to use gpu or not
    :return: the best available GPU device string
    """
    if torch.cuda.device_count() > 0 and use_gpu:
        device = "cuda"
    elif torch.backends.mps.is_available() and use_gpu and GLOBAL_ENABLE_MPS:
        device = "mps"
    else:
        device = "cpu"
    return device


def _get_ckpt_path(model_type, use_small=False):
    """
    Returns the path to the checkpoint file
    :param model_type: type of model (text, coarse, fine)
    :param use_small: whether to use small models or not
    :return: path to the checkpoint file
    """
    key = model_type
    if use_small or USE_SMALL_MODELS:
        key += "_small"
    return LOCAL_MODEL_PATHS[key]


def _download(from_hf_path, file_name):
    """
    Downloads the file from the given URL
    :param from_hf_path: path to the repo
    :param file_name: name of the file
    :return: None
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    hf_hub_download(repo_id=from_hf_path, filename=file_name, local_dir=CACHE_DIR)


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
    """
    Clears the CUDA cache
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def clean_models(model_key=None):
    """
    Cleans the models cache
    """
    global models
    model_keys = [model_key] if model_key is not None else list(models.keys())
    for k in model_keys:
        if k in models:
            del models[k]
    _clear_cuda_cache()
    gc.collect()


def _load_model(ckpt_path, device, use_small=False, model_type="text"):
    """
    Loads the model from the given path
    :param ckpt_path: path to the checkpoint file
    :param device: device to load the model
    :param use_small: whether to use small models or not
    :param model_type: type of model (text, coarse, fine)
    :return: the loaded model dict
    """
    if model_type == "text":
        ConfigClass = GPTConfig
        ModelClass = GPT
    elif model_type == "coarse":
        ConfigClass = GPTConfig
        ModelClass = GPT_COARSE
    elif model_type == "fine":
        ConfigClass = FineGPTConfig
        ModelClass = FineGPT
    else:
        raise NotImplementedError()
    model_key = f"{model_type}_small" if use_small or USE_SMALL_MODELS else model_type
    model_info = REMOTE_MODEL_PATHS[model_key]
    if not os.path.exists(ckpt_path):
        logger.info(f"{model_type} model not found, downloading into `{CACHE_DIR}`.")
        _download(model_info["repo_id"], model_info["file_name"])
    checkpoint = torch.load(ckpt_path, map_location=device)
    if model_type == "coarse":
        checkpoint["model"]['_orig_mod.lm_head.weight'] = checkpoint["model"]['_orig_mod.lm_head.weight'][
                                                          SEMANTIC_VOCAB_SIZE:, :]
    # this is a hack
    model_args = checkpoint["model_args"]
    if "input_vocab_size" not in model_args:
        model_args["input_vocab_size"] = model_args["vocab_size"]
        model_args["output_vocab_size"] = model_args["vocab_size"]
        del model_args["vocab_size"]
    gptconf = ConfigClass(**checkpoint["model_args"])
    model = ModelClass(gptconf)
    state_dict = checkpoint["model"]
    # fixup checkpoint
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    extra_keys = set(state_dict.keys()) - set(model.state_dict().keys())
    extra_keys = set([k for k in extra_keys if not k.endswith(".attn.bias")])
    missing_keys = set(model.state_dict().keys()) - set(state_dict.keys())
    missing_keys = set([k for k in missing_keys if not k.endswith(".attn.bias")])
    if len(extra_keys) != 0:
        raise ValueError(f"extra keys found: {extra_keys}")
    if len(missing_keys) != 0:
        raise ValueError(f"missing keys: {missing_keys}")
    model.load_state_dict(state_dict, strict=False)
    n_params = model.get_num_params()
    val_loss = checkpoint["best_val_loss"].item()
    logger.info(f"model loaded: {round(n_params / 1e6, 1)}M params, {round(val_loss, 3)} loss")
    model.eval()
    model.to(device)
    del checkpoint, state_dict
    _clear_cuda_cache()
    if model_type == "text":
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        total_config = yaml.safe_load(open('semantic_config.yaml', 'rb')) if use_small else yaml.safe_load(
            open('semantic_config_large.yaml', 'rb'))
        kv_gpt2_engine = TRTEngineFile(total_config['kv_engine_path'])
        kv_gpt2_trt = GPT2TRTDecoder(
            kv_gpt2_engine, total_config['GPT2_VARIANT'], gptconf, batch_size=total_config['batch_size']
        )
        return {
            "model": model,
            "tokenizer": tokenizer,
            "trt_model": kv_gpt2_trt,
        }
    if model_type == "coarse":
        gptconf.vocab_size = 2096
        total_config = yaml.safe_load(open('coarse_config.yaml', 'rb'))
        kv_gpt2_engine = TRTEngineFile(total_config['kv_engine_path'])
        kv_gpt2_trt = GPT2TRTDecoder(
            kv_gpt2_engine, total_config['GPT2_VARIANT'], gptconf, batch_size=total_config['batch_size']
        )
        return {
            "model": model,
            "trt_model": kv_gpt2_trt,
        }
    return model


def _load_vocos_model(device):
    """
    load vocos model
    :param device: device to load model
    :return: vocos model
    """
    model = Vocos.from_pretrained("charactr/vocos-encodec-24khz")
    model.eval()
    model.to(device)
    _clear_cuda_cache()
    return model


def load_model(use_gpu=True, use_small=False, force_reload=False, model_type="text"):
    """
    High level model loading function
    """
    _load_model_f = funcy.partial(_load_model, model_type=model_type, use_small=use_small)
    if model_type not in ("text", "coarse", "fine"):
        raise NotImplementedError()
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    model_key = f"{model_type}"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        ckpt_path = _get_ckpt_path(model_type, use_small=use_small)
        clean_models(model_key=model_key)
        model = _load_model_f(ckpt_path, device)
        models[model_key] = model
    if model_type == "text":
        models[model_key]["model"].to(device)
    elif model_type == "coarse":
        models[model_key]["model"].to(device)
    else:
        models[model_key].to(device)
    return models[model_key]


def load_vocos_model(use_gpu=True, force_reload=False):
    """
    High level vocos model loading function
    """
    global models
    global models_devices
    device = _grab_best_device(use_gpu=use_gpu)
    if device == "mps":
        # encodec doesn't support mps
        device = "cpu"
    model_key = "vocos"
    if OFFLOAD_CPU:
        models_devices[model_key] = device
        device = "cpu"
    if model_key not in models or force_reload:
        clean_models(model_key=model_key)
        model = _load_vocos_model(device)
        models[model_key] = model
    models[model_key].to(device)
    return models[model_key]


def preload_models(
        text_use_gpu=True,
        text_use_small=False,
        coarse_use_gpu=True,
        coarse_use_small=True,
        fine_use_gpu=True,
        fine_use_small=False,
        codec_use_gpu=True,
        force_reload=False,
):
    """Load all the necessary models for the pipeline."""
    if _grab_best_device() == "cpu" and (
            text_use_gpu or coarse_use_gpu or fine_use_gpu or codec_use_gpu
    ):
        logger.warning("No GPU being used. Careful, inference might be very slow!")
    _ = load_model(
        model_type="text", use_gpu=text_use_gpu, use_small=text_use_small, force_reload=force_reload
    )
    _ = load_model(
        model_type="coarse",
        use_gpu=coarse_use_gpu,
        use_small=coarse_use_small,
        force_reload=force_reload,
    )
    _ = load_model(
        model_type="fine", use_gpu=fine_use_gpu, use_small=fine_use_small, force_reload=force_reload
    )
    _ = load_vocos_model(use_gpu=codec_use_gpu, force_reload=force_reload)


####
# Generation Functionality
####


def _tokenize(tokenizer, text):
    return tokenizer.encode(text, add_special_tokens=False)


def _detokenize(tokenizer, enc_text):
    return tokenizer.decode(enc_text)


def _normalize_whitespace(text):
    return re.sub(r"\s+", " ", text).strip()


TEXT_ENCODING_OFFSET = 10_048
SEMANTIC_PAD_TOKEN = 10_000
TEXT_PAD_TOKEN = 129_595
SEMANTIC_INFER_TOKEN = 129_599


def _load_history_prompt(history_prompt_input):
    if isinstance(history_prompt_input, str) and history_prompt_input.endswith(".npz"):
        history_prompt = np.load(history_prompt_input)
    elif isinstance(history_prompt_input, str):
        # make sure this works on non-ubuntu
        history_prompt_input = os.path.join(*history_prompt_input.split("/"))
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
        top_k=None,
        top_p=None,
        silent=False,
        min_eos_p=0.2,
        max_gen_duration_s=None,
        allow_early_stop=True,
        use_kv_caching=False,
):
    """
    Generate semantic tokens from text in streaming.

    :param text: text to be converted to semantic tokens
    :param history_prompt: path to voice npz
    :param temp: temperature for semantic token generation
    :param top_k: select candidate tokens in top k for next semantic token generation
    :param top_p: select candidate tokens in p percent for next semantic token generation
    :param silent: disable progress bar
    :param min_eos_p: end of sentence token probability
            stop generating next semantic token if eos probability exceeds this value
    :param max_gen_duration_s: maximum length of generated next semantic token
    :param allow_early_stop: stop generating if next token is eos
    :param use_kv_caching: enable key value caching
    :return: generator that yields (chunks of semantic tokens, is_finished)
            first chunk is 27 + 20 * CHUNK_SIZE
            next chunks are appended 20 * CHUNK_SIZE
    """
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
    global models
    global models_devices
    if "text" not in models:
        preload_models()
    model_container = models["text"]
    model = model_container["model"]
    trt_model = model_container["trt_model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(_tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
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
    assert x.shape[1] == 256 + 256 + 1
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
                logits = trt_model(input_ids=x_input).logits
                # logits, kv_cache = model(x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache)
            else:
                x_input = x
                # key, values cache generation for context running
                logits, kv_cache = model(x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache)
                trt_model.load_past_key_values(kv_cache)
                trt_model.context_mode = False
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                original_device = relevant_logits.device
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(original_device)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(relevant_logits / temp, dim=-1)
            item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
            # item_next = torch.argmax(probs).unsqueeze(0).to(torch.int32)
            if allow_early_stop and (
                    item_next == SEMANTIC_VOCAB_SIZE
                    or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(n - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            # return results by 20 * CHUNK_SIZE because coarse generation consumes 20 tokens each time
            if n % (CHUNK_SIZE * 20) == 27 and n > 20 * CHUNK_SIZE:
                yield x.detach().cpu().numpy().squeeze()[256 + 256 + 1:], False
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
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1:]
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    _clear_cuda_cache()
    yield out, True


def _flatten_codebooks(arr, offset_size=CODEBOOK_SIZE):
    assert len(arr.shape) == 2
    arr = arr.copy()
    if offset_size is not None:
        for n in range(1, arr.shape[0]):
            arr[n, :] += offset_size * n
    flat_arr = arr.ravel("F")
    return flat_arr


COARSE_SEMANTIC_PAD_TOKEN = 12_048
COARSE_INFER_TOKEN = 12_050


def generate_coarse(
        x_semantic,
        is_finished,
        history_prompt=None,
        temp=0.7,
        top_k=None,
        top_p=None,
        silent=False,
        max_coarse_history=630,  # min 60 (faster), max 630 (more context)
        sliding_window_len=60,
        use_kv_caching=False,
        initial_n_step=0,
        initial_x_coarse_in=None
):
    """
    Generate coarse audio codes from semantic tokens.
    :param x_semantic: semantic tokens
    :param is_finished: is semantic token generation finished
    :param history_prompt: path to voice npz file
    :param temp: temperature for coarse generation
    :param top_k: select candidate tokens in top k for next coarse token generation
    :param top_p: select candidate tokens in p percent for next semantic token generation
    :silent: disable progress bar
    :param max_coarse_history: maximum number of coarse tokens from history is used for generation
    :param sliding_window_len: window length for coarse generation
    :param use_kv_caching: enable kv caching
    :param initial_n_step: initial number of steps for coarse generation
            this parameter is to resume coarse generation from previous step in streaming
    :param initial_x_coarse_in: initial coarse tokens from previous step in streaming
            this parameter should be set if initial_n_step is not 0

    :return gen_coarse_audio_arr: generated coarse tokens
            these tokens will be used for fine token generation
    :return x_coarse_in: coarse tokens from generating process
            these tokens are going to be used to resume coarse token generation
    :return n_step: number of steps in coarse generation
            this is going to be used to resume coarse token generation
    """
    assert (
            isinstance(x_semantic, np.ndarray)
            and len(x_semantic.shape) == 1
            and len(x_semantic) > 0
            and x_semantic.min() >= 0
            and x_semantic.max() <= SEMANTIC_VOCAB_SIZE - 1
    )
    assert 60 <= max_coarse_history <= 630
    assert max_coarse_history + sliding_window_len <= 1024 - 256
    semantic_to_coarse_ratio = COARSE_RATE_HZ / SEMANTIC_RATE_HZ * N_COARSE_CODEBOOKS
    max_semantic_history = int(np.floor(max_coarse_history / semantic_to_coarse_ratio))
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_semantic_history = history_prompt["semantic_prompt"]
        x_coarse_history = history_prompt["coarse_prompt"]
        assert (
                isinstance(x_semantic_history, np.ndarray)
                and len(x_semantic_history.shape) == 1
                and len(x_semantic_history) > 0
                and x_semantic_history.min() >= 0
                and x_semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
                and isinstance(x_coarse_history, np.ndarray)
                and len(x_coarse_history.shape) == 2
                and x_coarse_history.shape[0] == N_COARSE_CODEBOOKS
                and x_coarse_history.shape[-1] >= 0
                and x_coarse_history.min() >= 0
                and x_coarse_history.max() <= CODEBOOK_SIZE - 1
                and (
                        round(x_coarse_history.shape[-1] / len(x_semantic_history), 1)
                        == round(semantic_to_coarse_ratio / N_COARSE_CODEBOOKS, 1)
                )
        )
        x_coarse_history = _flatten_codebooks(x_coarse_history) + SEMANTIC_VOCAB_SIZE
        # trim histories correctly
        n_semantic_hist_provided = np.min(
            [
                max_semantic_history,
                len(x_semantic_history) - len(x_semantic_history) % 2,
                int(np.floor(len(x_coarse_history) / semantic_to_coarse_ratio)),
            ]
        )
        n_coarse_hist_provided = int(round(n_semantic_hist_provided * semantic_to_coarse_ratio))
        x_semantic_history = x_semantic_history[-n_semantic_hist_provided:].astype(np.int32)
        x_coarse_history = x_coarse_history[-n_coarse_hist_provided:].astype(np.int32)
        # TODO: bit of a hack for time alignment (sounds better)
        x_coarse_history = x_coarse_history[:-2]
    else:
        x_semantic_history = np.array([], dtype=np.int32)
        x_coarse_history = np.array([], dtype=np.int32)
    # load models if not yet exist
    global models
    global models_devices
    if "coarse" not in models:
        preload_models()
    model_container = models["coarse"]
    model = model_container["model"]
    trt_model = model_container["trt_model"]
    if OFFLOAD_CPU:
        model.to(models_devices["coarse"])
    device = next(model.parameters()).device
    # n_step initialization
    # if semantic generation is finished, set the ending step for coarse generation
    # else, set it infinity
    if is_finished:
        n_steps = int(
            round(
                np.floor(len(x_semantic) * semantic_to_coarse_ratio / N_COARSE_CODEBOOKS)
                * N_COARSE_CODEBOOKS
            )
        )
    else:
        n_steps = 1e4
    assert n_steps > 0 and n_steps % N_COARSE_CODEBOOKS == 0
    x_semantic = np.hstack([x_semantic_history, x_semantic]).astype(np.int32)
    x_coarse = x_coarse_history.astype(np.int32)
    base_semantic_idx = len(x_semantic_history)
    with _inference_mode():
        x_semantic_in = torch.from_numpy(x_semantic)[None].to(device)
        x_coarse_in = torch.from_numpy(x_coarse)[None].to(device)
        if initial_x_coarse_in is not None:
            x_coarse_in = initial_x_coarse_in
        n_window_steps = int(np.ceil(n_steps / sliding_window_len))
        n_step = initial_n_step
        for i_win in tqdm.tqdm(range(n_window_steps), total=n_window_steps, disable=silent):
            semantic_idx = base_semantic_idx + int(round(n_step / semantic_to_coarse_ratio))
            # pad from right side
            x_in = x_semantic_in[:, np.max([0, semantic_idx - max_semantic_history]):]
            x_in = x_in[:, :256]
            x_in = F.pad(
                x_in,
                (0, 256 - x_in.shape[-1]),
                "constant",
                COARSE_SEMANTIC_PAD_TOKEN,
            )
            x_in = torch.hstack(
                [
                    x_in,
                    torch.tensor([COARSE_INFER_TOKEN])[None].to(device),
                    x_coarse_in[:, -max_coarse_history:],
                ]
            )
            kv_cache = None
            for _ in range(sliding_window_len):
                if n_step >= n_steps:
                    continue
                is_major_step = n_step % N_COARSE_CODEBOOKS == 0

                if use_kv_caching and kv_cache is not None:
                    x_input = x_in[:, [-1]]
                    logits = trt_model(input_ids=x_input).logits
                    # logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                else:
                    x_input = x_in
                    logits, kv_cache = model(x_input, use_cache=use_kv_caching, past_kv=kv_cache)
                    trt_model.load_past_key_values(kv_cache)
                    trt_model.context_mode = False
                logit_start_idx = (
                        SEMANTIC_VOCAB_SIZE + (1 - int(is_major_step)) * CODEBOOK_SIZE
                )
                logit_end_idx = (
                        SEMANTIC_VOCAB_SIZE + (2 - int(is_major_step)) * CODEBOOK_SIZE
                )
                relevant_logits = logits[
                                  0,
                                  0,
                                  logit_start_idx - SEMANTIC_VOCAB_SIZE:logit_end_idx - SEMANTIC_VOCAB_SIZE
                ]
                if top_p is not None:
                    # faster to convert to numpy
                    original_device = relevant_logits.device
                    relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                    sorted_indices = np.argsort(relevant_logits)[::-1]
                    sorted_logits = relevant_logits[sorted_indices]
                    cumulative_probs = np.cumsum(softmax(sorted_logits))
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                    sorted_indices_to_remove[0] = False
                    relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                    relevant_logits = torch.from_numpy(relevant_logits)
                    relevant_logits = relevant_logits.to(original_device)
                if top_k is not None:
                    v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                    relevant_logits[relevant_logits < v[-1]] = -float("Inf")
                probs = F.softmax(relevant_logits / temp, dim=-1)
                item_next = torch.multinomial(probs, num_samples=1).to(torch.int32)
                # item_next = torch.argmax(probs).unsqueeze(0).to(torch.int32)
                item_next += logit_start_idx
                x_coarse_in = torch.cat((x_coarse_in, item_next[None]), dim=1)
                x_in = torch.cat((x_in, item_next[None]), dim=1)
                del logits, relevant_logits, probs, item_next
                n_step += 1
            del x_in
            if not is_finished and i_win == CHUNK_SIZE - 1:
                break
        del x_semantic_in
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_coarse_arr = x_coarse_in.detach().cpu().numpy().squeeze()[len(x_coarse_history):]
    # del x_coarse_in
    # assert len(gen_coarse_arr) == n_steps
    gen_coarse_audio_arr = gen_coarse_arr.reshape(-1, N_COARSE_CODEBOOKS).T - SEMANTIC_VOCAB_SIZE
    for n in range(1, N_COARSE_CODEBOOKS):
        gen_coarse_audio_arr[n, :] -= n * CODEBOOK_SIZE
    _clear_cuda_cache()
    return gen_coarse_audio_arr, x_coarse_in, n_step


def generate_fine(
        x_coarse_gen,
        history_prompt=None,
        temp=0.5,
        silent=True,
):
    """
    Generate full audio codes from coarse audio codes.

    :param x_coarse_gen: coarse tokens
    :param history_prompt: file path to voice npz file
    :param temp: temperature for fine token generation
    :param silent: disable progress bar
    :return: fine tokens
    """
    assert (
            isinstance(x_coarse_gen, np.ndarray)
            and len(x_coarse_gen.shape) == 2
            and 1 <= x_coarse_gen.shape[0] <= N_FINE_CODEBOOKS - 1
            and x_coarse_gen.shape[1] > 0
            and x_coarse_gen.min() >= 0
            and x_coarse_gen.max() <= CODEBOOK_SIZE - 1
    )
    if history_prompt is not None:
        history_prompt = _load_history_prompt(history_prompt)
        x_fine_history = history_prompt["fine_prompt"]
        assert (
                isinstance(x_fine_history, np.ndarray)
                and len(x_fine_history.shape) == 2
                and x_fine_history.shape[0] == N_FINE_CODEBOOKS
                and x_fine_history.shape[1] >= 0
                and x_fine_history.min() >= 0
                and x_fine_history.max() <= CODEBOOK_SIZE - 1
        )
    else:
        x_fine_history = None
    n_coarse = x_coarse_gen.shape[0]
    # load models if not yet exist
    global models
    global models_devices
    if "fine" not in models:
        preload_models()
    model = models["fine"]
    if OFFLOAD_CPU:
        model.to(models_devices["fine"])
    device = next(model.parameters()).device
    # make input arr
    in_arr = np.vstack(
        [
            x_coarse_gen,
            np.zeros((N_FINE_CODEBOOKS - n_coarse, x_coarse_gen.shape[1]))
            + CODEBOOK_SIZE,  # padding
        ]
    ).astype(np.int32)
    # prepend history if available (max 512)
    if x_fine_history is not None:
        x_fine_history = x_fine_history.astype(np.int32)
        in_arr = np.hstack(
            [
                x_fine_history[:, -512:].astype(np.int32),
                in_arr,
            ]
        )
        n_history = x_fine_history[:, -512:].shape[1]
    else:
        n_history = 0
    n_remove_from_end = 0
    # need to pad if too short (since non-causal model)
    if in_arr.shape[1] < 1024:
        n_remove_from_end = 1024 - in_arr.shape[1]
        in_arr = np.hstack(
            [
                in_arr,
                np.zeros((N_FINE_CODEBOOKS, n_remove_from_end), dtype=np.int32) + CODEBOOK_SIZE,
            ]
        )
    # we can be lazy about fractional loop and just keep overwriting codebooks
    n_loops = np.max([0, int(np.ceil((x_coarse_gen.shape[1] - (1024 - n_history)) / 512))]) + 1
    with _inference_mode():
        in_arr = torch.tensor(in_arr.T).to(device)
        for n in tqdm.tqdm(range(n_loops), disable=silent):
            start_idx = np.min([n * 512, in_arr.shape[0] - 1024])
            start_fill_idx = np.min([n_history + n * 512, in_arr.shape[0] - 512])
            rel_start_fill_idx = start_fill_idx - start_idx
            in_buffer = in_arr[start_idx: start_idx + 1024, :][None]
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                logits = model(nn, in_buffer)
                if temp is None:
                    relevant_logits = logits[0, rel_start_fill_idx:, :CODEBOOK_SIZE]
                    codebook_preds = torch.argmax(relevant_logits, -1)
                else:
                    relevant_logits = logits[0, :, :CODEBOOK_SIZE] / temp
                    probs = F.softmax(relevant_logits, dim=-1)
                    codebook_preds = torch.multinomial(
                        probs[rel_start_fill_idx:1024], num_samples=1
                    ).reshape(-1)
                codebook_preds = codebook_preds.to(torch.int32)
                in_buffer[0, rel_start_fill_idx:, nn] = codebook_preds
                del logits, codebook_preds
            # transfer over info into model_in and convert to numpy
            for nn in range(n_coarse, N_FINE_CODEBOOKS):
                in_arr[
                    start_fill_idx: start_fill_idx + (1024 - rel_start_fill_idx), nn
                ] = in_buffer[0, rel_start_fill_idx:, nn]
            del in_buffer
        gen_fine_arr = in_arr.detach().cpu().numpy().squeeze().T
        del in_arr
    if OFFLOAD_CPU:
        model.to("cpu")
    gen_fine_arr = gen_fine_arr[:, n_history:]
    if n_remove_from_end > 0:
        gen_fine_arr = gen_fine_arr[:, :-n_remove_from_end]
    assert gen_fine_arr.shape[-1] == x_coarse_gen.shape[-1]
    _clear_cuda_cache()
    return gen_fine_arr


def vocos_decode(fine_tokens):
    """
    Turn quantized audio codes into audio array using vocos.
    """
    # load models if not yet exist
    global models
    global models_devices
    if "vocos" not in models:
        preload_models()
    model = models["vocos"]
    if OFFLOAD_CPU:
        model.to(models_devices["vocos"])
    device = next(model.parameters()).device
    audio_tokens_torch = torch.from_numpy(fine_tokens).to(device)
    features = model.codes_to_features(audio_tokens_torch)
    arr = model.decode(features, bandwidth_id=torch.tensor([2], device=device)).squeeze()
    del features
    if OFFLOAD_CPU:
        model.to("cpu")
    return arr
