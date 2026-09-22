"""Wrapper around the project's LLM SDK providing a Small_LLM_Model
adapter with convenience methods for encoding, decoding and accessing
tokenizer/vocab file paths.
"""

import numpy as np

from call_me_maybe.parsers import _errors
from llm_sdk import Small_LLM_Model


class LLModel:
    """A thin wrapper exposing a small LLM model from llm_sdk."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Create and load a Small_LLM_Model instance.

        Args:
            model_name: The model identifier to load (e.g. a Hugging Face
                repository path). Defaults to 'Qwen/Qwen3-0.6B'.

        Raises:
            _errors.LLmModelLoadError: If the model cannot be loaded due to
                OS errors, runtime/device issues, or configuration/import
                problems.
        """

        try:
            self.small_llm_model = Small_LLM_Model(
                model_name=model_name,
                device=None,  # docs says -> picks mps, then cuda, then cpu
                dtype=None,  # float32 or float16 for weights accurate
                trust_remote_code=True,  # allow custom code from model repo
            )
        except OSError as e:
            msg = f"Failed to load model: {e}"
            raise _errors.LLmModelLoadError(msg) from e
        except RuntimeError as e:
            msg = f"Device or CUDA error during model load: {e}"
            raise _errors.LLmModelLoadError(msg) from e
        except (ValueError, ImportError) as e:
            msg = f"Configuration error during model load: {e}"
            raise _errors.LLmModelLoadError(msg) from e

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into a tensor of token ids.

        Args:
            text: Input string to encode.

        Raises:
            _errors.LLmModelEncodeError: If encoding fails for any reason.

        Returns:
            A NumPy array containing token ids representing the input text.
        """

        try:
            encoded = self.small_llm_model.encode(text=text)
            return np.asarray(encoded.tolist(), dtype=np.int64)
        except Exception as e:
            msg = f"Unexpected error during encoding: {e}"
            raise _errors.LLmModelEncodeError(msg) from e

    def decode_text(self, token_ids_tensor: np.ndarray) -> str:
        """Decode a NumPy array of token ids back to a string.

        Args:
            token_ids_tensor: NumPy array containing token ids to decode.

        Raises:
            _errors.LLmModelDecodeError: If decoding fails for any reason.

        Returns:
            The decoded string.
        """

        try:
            return self.small_llm_model.decode(ids=token_ids_tensor.tolist())
        except Exception as e:
            msg = f"Unexpected error during decoding: {e}"
            raise _errors.LLmModelDecodeError(msg) from e

    def get_all_next_tokens_logits(
            self, token_ids_encoded: list[int]) -> list[float]:
        """Return logits for the next-token prediction given input ids.

        Args:
            token_ids_decoded: List of token ids representing the input
                sequence.

        Returns:
            A list or array-like of logits for the next-token prediction.
        """
        return self.small_llm_model.get_logits_from_input_ids(
            token_ids_encoded)

    def get_vocab_path(self) -> str:
        """Return the filesystem path to the model's vocabulary file.

        Returns:
            Path to the vocab file as a string.
        """
        return self.small_llm_model.get_path_to_vocab_file()
