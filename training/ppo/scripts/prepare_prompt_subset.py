#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import click

from training.ppo.pipeline_utils import (
    aligned_prompt_subset,
    parse_seed_spec,
    write_prompt_lines,
)


@click.command()
@click.option("--prompts", "prompt_path", type=click.Path(exists=True, dir_okay=False), required=True)
@click.option("--output", type=click.Path(dir_okay=False), required=True)
@click.option("--seeds", type=str, default=None, help="Seed list/range; count is derived from it.")
@click.option("--count", type=click.IntRange(min=1), default=None, help="Explicit prompt count when --seeds is not used.")
@click.option("--start", type=click.IntRange(min=0), default=0, show_default=True, help="Starting prompt index before wraparound.")
def main(prompt_path: str, output: str, seeds: Optional[str], count: Optional[int], start: int) -> None:
    if seeds is None and count is None:
        raise click.ClickException("Either --seeds or --count must be provided.")
    if seeds is not None and count is not None:
        raise click.ClickException("Use either --seeds or --count, not both.")

    resolved_count = count if count is not None else len(parse_seed_spec(seeds or ""))
    prompts = aligned_prompt_subset(prompt_path, count=resolved_count, start=start)
    output_path = write_prompt_lines(output, prompts)

    print(
        json.dumps(
            {
                "source_prompts": str(Path(prompt_path).expanduser().resolve()),
                "output_prompts": str(output_path),
                "count": resolved_count,
                "start": start,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
