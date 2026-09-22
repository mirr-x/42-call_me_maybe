"""Logits processing utilities for language model output."""

import numpy as np


class LogitsProcessor:
    """Processor for handling logits arrays from language models."""

    def __init__(self, logits: np.ndarray) -> None:

        self.logits: np.ndarray = logits

    def get_best_token(self) -> int:
        """Get the token ID with the highest logit score.

        Returns:
            int: The index of the token with the maximum logit value
        """

        return int(np.argmax(self.logits))

    def get_top_k_tokens(self, k: int) -> list[int]:
        """Get the token IDs with the top k highest logit scores.

        Args:
            k (int): Number of top tokens to retrieve

        Returns:
            list[int]: List of token IDs sorted by logit score in descend order
        """

        return np.argsort(self.logits)[-k:][::-1].tolist()

    def get_token_score(self, token_id: int) -> float:
        """Get the logit score for a specific token.

        Args:
            token_id (int): The ID of the token to retrieve the score for

        Returns:
            float: The logit score of the specified token
        """

        return float(self.logits[token_id])

    def mask_logits(self, allowed_tokens: set[int]) -> None:
        """Apply masking to tokens by setting logits for disallowed
            token IDs to -inf.

        Args:
            allowed_tokens (set[int]): Set of token IDs that remain unmasked.

        Returns:
            None: The logits tensor is modified in place.
        """

        mask = np.ones(self.logits.shape, dtype=bool)
        valid_tokens = [token_id for token_id in allowed_tokens
                        if 0 <= token_id < len(self.logits)]
        mask[valid_tokens] = False
        self.logits[mask] = -np.inf

    def block_token(self, token_id: int) -> None:
        """Block a specific token by setting its logit to -inf.

        Args:
            token_id (int): The ID of the token to block.
        """
        self.logits[token_id] = float("-inf")
