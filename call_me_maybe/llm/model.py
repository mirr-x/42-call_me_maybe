""" fill me """

import torch

from llm_sdk import llm_sdk
from call_me_maybe.parsers import _errors


class LLModel:
    """ fill me """

    def __init__(self, model_name: str = 'Qwen/Qwen3-0.6B') -> None:
        """ fill me """

        try:
            self.small_llm_model = llm_sdk.Small_LLM_Model(
                model_name=model_name,
                device=None,  # docs says -> picks mps, then cuda, then cpu
                dtype=None,  # float32 or float16 for weights accurate
                trust_remote_code=True,  # allow custom code from model repo
            )
        except OSError as e:
            msg = f'Failed to load model: {e}'
            raise _errors.LLmModelLoadError(msg) from e
        except RuntimeError as e:
            msg = f'Device or CUDA error during model load: {e}'
            raise _errors.LLmModelLoadError(msg) from e
        except (ValueError, ImportError) as e:
            msg = f'Configuration error during model load: {e}'
            raise _errors.LLmModelLoadError(msg) from e

    def encode_text(self, text: str) -> torch.Tensor:
        """ fill me """

        try:
            return self.small_llm_model.encode(text=text)
        except Exception as e:
            msg = f'Unexpected error during encoding: {e}'
            raise _errors.LLmModelEncodeError(msg) from e
