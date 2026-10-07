from __future__ import annotations

from typing import Any

import numpy as np


def resolve_device(device: str | None) -> str | None:
    if device in {None, "", "auto"}:
        try:
            import torch

            return "cuda:0" if torch.cuda.is_available() else "cpu"
        except Exception:
            return None
    return device


class BGEEncoder:
    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = "auto",
        use_fp16: bool | None = None,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        resolved_device = resolve_device(device)
        if use_fp16 is None:
            use_fp16 = bool(resolved_device and resolved_device.startswith("cuda"))
        self.model = BGEM3FlagModel(
            model_path,
            use_fp16=use_fp16,
            device=resolved_device,
        )

    def encode_dense(
        self,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> np.ndarray:
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.asarray(output["dense_vecs"], dtype=np.float32)



class BGEReranker:
    def __init__(
        self,
        model_path: str,
        *,
        device: str | None = "auto",
        use_fp16: bool | None = None,
    ) -> None:
        from FlagEmbedding import FlagReranker

        resolved_device = resolve_device(device)
        if use_fp16 is None:
            use_fp16 = bool(resolved_device and resolved_device.startswith("cuda"))
        self.model = FlagReranker(model_path, use_fp16=use_fp16, device=resolved_device)

    def score(
        self,
        query: str,
        texts: list[str],
        *,
        batch_size: int,
        max_length: int,
    ) -> list[float]:
        if not texts:
            return []
        pairs: list[tuple[str, str]] = [(query, text) for text in texts]
        scores: Any = self.model.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=False,
        )
        if isinstance(scores, (float, int)):
            return [float(scores)]
        return [float(score) for score in scores]
