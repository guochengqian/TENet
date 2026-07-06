"""Command-line helper for the tenet_official package.

Prints TENet resource links (paper, code, PixelShift200 dataset) so pip users
can quickly find where to get the code, data, and how weights are loaded.
"""

from __future__ import annotations

import argparse

from . import __version__
from .info import MODELS, RESOURCES, REPOSITORY


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TENet resource/info helper (paper, code, PixelShift200 dataset)."
    )
    parser.add_argument("--models", action="store_true", help="List available model variants and exit.")
    parser.add_argument("--version", action="store_true", help="Print version and exit.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        print(__version__)
        return 0

    if args.models:
        print("TENet model variants (model/<name>.py, built as model.<name>.Net(args)):")
        for name in MODELS:
            print(f"  - {name}")
        return 0

    print(f"TENet (tenet_official {__version__})")
    print("Rethinking Learning-based Demosaicing, Denoising, and Super-Resolution Pipeline\n")
    for res in RESOURCES.values():
        print(f"  {res.key:14s} {res.url}\n  {'':14s} {res.description}")
    print(
        "\nTraining/evaluation uses a source checkout — see "
        f"{REPOSITORY} (`source install.sh`).\n"
        "Pretrained weights are passed via `--pretrain /path/to/checkpoint.pth`."
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
