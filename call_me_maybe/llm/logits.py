"""Logits processing utilities for language model output."""

import torch


class LogitsProcessor:
    """Processor for handling logits arrays from language models."""

    def __init__(self, logits: torch.Tensor) -> None:

        self.logits: torch.Tensor = logits

    def get_best_token(self) -> int:
        """Get the token ID with the highest logit score.

        Returns:
            int: The index of the token with the maximum logit value
        """

        return torch.argmax(self.logits).item()

    def get_top_k_tokens(self, k: int) -> list[int]:
        """Get the token IDs with the top k highest logit scores.

        Args:
            k (int): Number of top tokens to retrieve

        Returns:
            list[int]: List of token IDs sorted by logit score in descend order
        """

        _, top_k_indices = torch.topk(self.logits, k)

        return top_k_indices.tolist()

    def get_token_score(self, token_id: int) -> float:
        """Get the logit score for a specific token.

        Args:
            token_id (int): The ID of the token to retrieve the score for

        Returns:
            float: The logit score of the specified token
        """

        return self.logits[token_id].item()

    def mask_logits(self, allowed_tokens: set[int]) -> None:
        """Apply masking to tokens by setting logits for disallowed
            token IDs to -inf.

        Args:
            allowed_tokens (set[int]): Set of token IDs that remain unmasked.

        Returns:
            None: The logits tensor is modified in place.
        """

        allowed = self.logits.new_tensor(
            list(allowed_tokens),
            dtype=torch.long
        )

        mask = torch.ones_like(self.logits, dtype=torch.bool)
        mask[allowed] = False

        self.logits.masked_fill_(mask, float("-inf"))
