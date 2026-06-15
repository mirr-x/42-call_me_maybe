"""Logits processing utilities for language model output."""

import numpy as np


class LogitsProcessor:
    """Processor for handling logits arrays from language models."""

    def __init__(self, logits: list[float]) -> None:

        self.logits = np.array(logits)

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

        top_k_indices = np.argsort(self.logits)[::-1][:k]

        return top_k_indices.tolist()

    def get_token_score(self, token_id: int) -> float:
        """Get the logit score for a specific token.

        Args:
            token_id (int): The ID of the token to retrieve the score for

        Returns:
            float: The logit score of the specified token
        """

        return self.logits[token_id]

    def mask_tokens(self, invalid_token_ids: list[int]) -> np.ndarray:
        """Apply masking to invalid tokens by setting their logits to -inf.

        Args:
            invalid_token_ids (list[int]): List of token IDs to mask

        Returns:
            np.ndarray: New logits array with masked tokens set to -inf
        """

        logits_copy = self.logits.copy()
        vocab_size = logits_copy.shape[0]
        valid_token_ids = [
            token_id
            for token_id in invalid_token_ids
            if isinstance(token_id, (int, np.integer)) and
            0 <= token_id < vocab_size
        ]

        if valid_token_ids:
            for id_ in valid_token_ids:
                logits_copy[id_] = float("-inf")
        return logits_copy
