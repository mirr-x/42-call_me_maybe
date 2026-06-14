"""LLM SDK for local model inference using Hugging Face transformers.

Provides Small_LLM_Model class for loading and running causal language
models.
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    logging,
)
from huggingface_hub import hf_hub_download


logging.set_verbosity_error()  # keep the console clean


class Small_LLM_Model:
    """Small helper around Hugging Face causal language models.

    It loads a pretrained model and tokenizer, then gives you a few simple
    methods for turning text into tokens, turning tokens back into text, and
    reading the model's next-token logits.

    Args:
        model_name (str): Hugging Face model id.
            Defaults to ``"Qwen/Qwen3-0.6B"``.
        device (str | None): Where to run the model.
            If ``None``, picks ``mps``, then ``cuda``, then ``cpu``.
        dtype (torch.dtype | None): Number type for model weights.
            Uses ``float16`` on GPU/MPS and ``float32`` on CPU.
        trust_remote_code (bool): Allow custom code from the model repo.
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

        # Pick the best available device automatically.
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        self._device = device

        if dtype is None:
            dtype = (
                torch.float16
                if self._device in ["cuda", "mps"]
                else torch.float32
            )
        self._dtype = dtype

        # --- load tokenizer & model -----------------------------------------
        self._tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            # Some tokenizers do not define a pad token, so reuse EOS.
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            device_map=(
                "auto" if self._device == "cuda" else None
            ),
            trust_remote_code=trust_remote_code,
        )
        self._model.to(self._device)
        self._model.eval()

        # Make sure the model is used only for inference.
        for p in self._model.parameters():
            p.requires_grad = False

    def encode(self, text: str) -> torch.Tensor:
        """Turn text into token token_ids.

        Args:
            text (str): Input text.

        Returns:
            torch.Tensor: A 2-D tensor with shape ``(1, sequence_length)``.
        """
        token_ids = self._tokenizer.encode(text, add_special_tokens=False)
        return torch.tensor([token_ids], device=self._device, dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        """Turn token ids back into text.

        Args:
            ids (torch.Tensor | list[int]): Token ids as a list or tensor.

        Returns:
            str: Decoded text without special tokens.
        """
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return str(self._tokenizer.decode(ids, skip_special_tokens=True))

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        """Get the raw next-token scores for a sequence.

        The model runs without gradients, and the last token position is used.

        Args:
            input_ids (list[int]): Token ids for one input sequence.

        Returns:
            list[float]: Raw logits for the next token.
        """
        input_tensor = torch.tensor(
            [input_ids],
            device=self._device,
            dtype=torch.long
        )
        with torch.no_grad():
            out = self._model(input_ids=input_tensor)
        # Use the last token to predict the next one.
        logits = out.logits[0, -1, -1].tolist()
        return [float(x) for x in logits]

    def get_path_to_vocab_file(self) -> str:
        """Get the local path to the tokenizer vocab file.

        Returns:
            str: Local path to the downloaded vocab file.
        """
        vocab_file_name = self._tokenizer.vocab_files_names.get(
            "vocab_file",
            "vocab.json",
        )
        vocab_path = hf_hub_download(
            repo_id=self._model_name,
            filename=vocab_file_name,
        )
        return str(vocab_path)

    def get_path_to_merges_file(self) -> str:
        """Get the local path to the tokenizer merges file.

        Returns:
            str: Local path to the merges file, if the tokenizer uses one.
        """
        merges_file_name = self._tokenizer.vocab_files_names.get(
            "merges_file",
            "merges.txt",
        )
        merges_path = hf_hub_download(
            repo_id=self._model_name,
            filename=merges_file_name,
        )
        return str(merges_path)

    def get_path_to_tokenizer_file(self) -> str:
        """Get the local path to the tokenizer JSON file.

        Returns:
            str: Local path to the tokenizer JSON file.
        """
        tokenizer_file_name = self._tokenizer.vocab_files_names.get(
            "tokenizer_file",
            "tokenizer.json",
        )
        tokenizer_path = hf_hub_download(
            repo_id=self._model_name,
            filename=tokenizer_file_name,
        )
        return str(tokenizer_path)
