"""Chunk selection helpers for distributed simulation runs."""

from __future__ import annotations

from typing import TypeVar


T = TypeVar("T")


__all__ = ["select_design_chunk", "validate_chunk_args"]


def validate_chunk_args(chunk_index: int | None, num_chunks: int | None) -> None:
    """Validate paired chunking arguments."""
    if (chunk_index is None) != (num_chunks is None):
        raise ValueError("--chunk-index and --num-chunks must be provided together")
    if chunk_index is None or num_chunks is None:
        return
    if not isinstance(chunk_index, int) or isinstance(chunk_index, bool):
        raise ValueError("--chunk-index must be an integer")
    if not isinstance(num_chunks, int) or isinstance(num_chunks, bool):
        raise ValueError("--num-chunks must be an integer")
    if num_chunks < 1:
        raise ValueError("--num-chunks must be at least 1")
    if chunk_index < 0 or chunk_index >= num_chunks:
        raise ValueError("--chunk-index must satisfy 0 <= chunk_index < num_chunks")


def select_design_chunk(
    designs: list[T],
    chunk_index: int | None,
    num_chunks: int | None,
) -> list[T]:
    """Select one deterministic strided chunk of designs."""
    validate_chunk_args(chunk_index, num_chunks)
    if chunk_index is None or num_chunks is None:
        return list(designs)
    return list(designs[chunk_index::num_chunks])
