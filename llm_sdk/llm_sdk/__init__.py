# ABOUTME: LLM SDK for local model inference using Hugging Face transformers.
# ABOUTME: Provides Small_LLM_Model class for loading and running causal language models.

import time
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel, logging
from huggingface_hub import hf_hub_download
import os


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Lightweight wrapper for Hugging Face causal language models.

    This class simplifies loading a pretrained causal language model and its tokenizer from
    the Hugging Face Hub and exposes small helper methods for tokenisation, decoding and
    obtaining raw logits for the next token. It is designed for fast, low-memory experiments.

    Args:
        model_name (str): Hugging Face repository identifier for the model. Defaults to
            ``"Qwen/Qwen3-0.6B"``.
        device (str | None): Computation device. If ``None``, the class automatically
            selects ``mps`` (macOS) or ``cuda`` when available, otherwise ``cpu``.
        dtype (torch.dtype | None): Numeric precision for model weights. Defaults to
            ``torch.float16`` on GPU/MPS and ``torch.float32`` on CPU when unspecified.
        trust_remote_code (bool): Whether to allow and execute remote model code from the
            model repository.

    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        *,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        trust_remote_code: bool = True,
    ) -> None:
        self._model_name = model_name

        # Auto-select device with priority: mps > cuda > cpu
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = torch.float16 if self._device in ["cuda", "mps"] else torch.float32
        self._dtype = dtype

        # --- load tokenizer & model -------------------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self._tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map="auto" if self._device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # switch to inference-only mode
        for p in self._model.parameters():
            p.requires_grad = False


    def encode(self, text: str) -> torch.Tensor:
        """Tokenize an input string and return input IDs as a tensor on the model device.

        Args:
            text (str): The text to tokenize.

        Returns:
            torch.Tensor: A 2-D tensor with shape ``(1, sequence_length)`` of type
                ``torch.long`` placed on the configured device.
        """
        ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([ids], device=self._device, dtype=torch.long)


    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Convert token ids back to a readable string, removing special tokens.

        Args:
            ids (torch.Tensor | list[int]): Token ids as a 1-D list or a tensor. If a
                tensor is provided it will be converted to a Python list.

        Returns:
            str: Decoded string with special tokens removed.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return self._tokenizer.decode(ids, skip_special_tokens=True)


    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Return raw logits (pre-softmax) for the next token given input ids.

        This runs the model in evaluation mode with gradients disabled. The method returns
        the logits vector for the last position in the provided sequence.

        Args:
            input_ids (list[int]): A list of token ids representing a single input sequence.

        Returns:
            list[float]: A Python list containing the raw logits for the next token.
        """
        input_tensor = torch.tensor([input_ids], device=self._device, dtype=torch.long)
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Get logits for the last token in the sequence for the batch (batch size 1)
        logits = out.logits[0, -1].tolist()
        return [float(x) for x in logits]


    def get_path_to_vocab_file(self) -> str:
        """Download and return the local path to the tokenizer vocab file.

        Returns:
            str: Local filesystem path to the downloaded vocab file for the tokenizer.
        """
        vocab_file_name = self._tokenizer.vocab_files_names.get('vocab_file', "vocab.json")
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name
        )
        return vocab_path


    def get_path_to_merges_file(self) -> str:
        """Download and return the local path to the tokenizer merges file.

        Returns:
            str: Local filesystem path to the downloaded merges file (if applicable).
        """
        merges_file_name = self._tokenizer.vocab_files_names.get('merges_file', "merges.txt")
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name
        )
        return merges_path


    def get_path_to_tokenizer_file(self) -> str:
        """Download and return the local path to the tokenizer JSON file.

        Returns:
            str: Local filesystem path to the tokenizer JSON file used by the tokenizer.
        """
        tokenizer_file_name = self._tokenizer.vocab_files_names.get('tokenizer_file', "tokenizer.json")
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name
        )
        return tokenizer_path
