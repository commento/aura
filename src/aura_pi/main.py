from __future__ import annotations

import argparse

from .config import load_config
from .pipeline import AuraPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aura Pi realtime pipeline")
    parser.add_argument("--config", required=True, help="Percorso del file YAML di configurazione")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    pipeline = AuraPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
