"""TENet release helper package.

TENet (ICCP'22) — *Rethinking Learning-based Demosaicing, Denoising, and
Super-Resolution Pipeline*. The research training/evaluation code lives in the
TENet repository. This small package provides stable metadata, resource links,
and helpers for pip-installed users (paper, PixelShift200 dataset, pretrained
weights, and the model list).
"""

from .info import (
    ARXIV_URL,
    PIXELSHIFT200,
    PROJECT_PAGE,
    REPOSITORY,
    MODELS,
    RESOURCES,
    resources,
)

__all__ = [
    "ARXIV_URL",
    "PIXELSHIFT200",
    "PROJECT_PAGE",
    "REPOSITORY",
    "MODELS",
    "RESOURCES",
    "resources",
]

__version__ = "0.1.0"
