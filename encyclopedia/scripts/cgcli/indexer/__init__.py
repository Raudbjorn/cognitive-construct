"""Codegraph indexer — embeds CGC's graph-building engine."""

from .config import IndexConfig
from .graph_builder import GraphBuilder

__all__ = ["GraphBuilder", "IndexConfig"]
