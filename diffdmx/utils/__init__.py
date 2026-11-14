"""Minimal dynamic-import utility used by the model configurations."""

from importlib import import_module


def get_hook(import_path: str):
    module_path, attribute = import_path.rsplit(".", 1)
    return getattr(import_module(module_path), attribute)
