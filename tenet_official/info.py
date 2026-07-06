"""Stable metadata and resource links for TENet.

TENet is a research codebase (training/evaluation scripts), not a library, so
this package intentionally ships metadata + resource pointers rather than
importable model code. Use a source checkout of the repository to train/evaluate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

ARXIV_URL = "https://arxiv.org/abs/1905.02538"
PROJECT_PAGE = "http://guochengqian.github.io/project/pixelshift200"
REPOSITORY = "https://github.com/guochengqian/TENet"


@dataclass(frozen=True)
class Resource:
    """A named external resource with a URL and short description."""

    key: str
    url: str
    description: str


# PixelShift200 dataset. The .mat files are 4-channel (R, Gr, Gb, B) full-color
# captures produced by pixel-shift photography; original RAW (.ARW) files are
# also provided. See the project page for details.
PIXELSHIFT200: Dict[str, str] = {
    "project_page": PROJECT_PAGE,
    "google_drive": "https://drive.google.com/drive/folders/1S1mt9D8IlpSAVm7RTjLheh1hTaltsV_f?usp=share_link",
}


# TENet model variants available in the `model/` package of the repository
# (built via `importlib.import_module("model.<name>").Net(args)`).
MODELS: List[str] = [
    "tenet",
    "tenetv0",
    "tenetv2",
    "tenetv3",
    "jdndmsr",
    "jdndmsr_e2e",
    "jdsr",
    "resnet",
]


RESOURCES: Dict[str, Resource] = {
    "paper": Resource("paper", ARXIV_URL, "TENet paper (arXiv:1905.02538, ICCP'22)"),
    "code": Resource("code", REPOSITORY, "Training/evaluation source code"),
    "project": Resource("project", PROJECT_PAGE, "PixelShift200 project page"),
    "pixelshift200": Resource(
        "pixelshift200",
        PIXELSHIFT200["google_drive"],
        "PixelShift200 dataset + original RAW files (Google Drive)",
    ),
}


def resources() -> Dict[str, Resource]:
    """Return the known TENet resources keyed by short name."""

    return dict(RESOURCES)
