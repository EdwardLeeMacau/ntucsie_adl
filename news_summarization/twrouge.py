import os

from ckiptagger import WS, data_utils
from rouge import Rouge

cache_dir = os.environ.get("XDG_CACHE_HOME", os.path.join(os.getenv("HOME"), ".cache"))
download_dir = os.path.join(cache_dir, "ckiptagger")
data_dir = os.path.join(cache_dir, "ckiptagger/data")
os.makedirs(download_dir, exist_ok=True)
if not os.path.exists(os.path.join(data_dir, "model_ws")):
    data_utils.download_data_gdown(download_dir)

# word segmenting
ws = WS(data_dir)


def tokenize_and_join(sentences):
    return [" ".join(toks) for toks in ws(sentences)]


rouge = Rouge()


def get_rouge(predictions, refs, avg=True, ignore_empty=False):
    """wrapper around: from rouge import Rouge
    Args:
        predictions: string or list of strings
        refs: string or list of strings
        avg: bool, return the average metrics if set to True
        ignore_empty: bool, ignore empty pairs if set to True
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(refs, list):
        refs = [refs]

    predictions, refs = tokenize_and_join(predictions), tokenize_and_join(refs)

    try:
        scores = rouge.get_scores(predictions, refs, avg=avg, ignore_empty=ignore_empty)
    except ValueError:
        # Workaround of ValueError: Hypothesis is empty.
        scores = {
            'rouge-1': { 'r': 0.0, 'p': 0.0, 'f': 0.0 },
            'rouge-2': { 'r': 0.0, 'p': 0.0, 'f': 0.0 },
            'rouge-l': { 'r': 0.0, 'p': 0.0, 'f': 0.0 }
        }

    return scores
