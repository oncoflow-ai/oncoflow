"""
cli.py – OncoFlow inference CLI (Typer).

Subcommands:
    oncoflow-infer segment       – 3-model panel on one NIfTI
    oncoflow-infer longitudinal  – full comparison between two NIfTIs
    oncoflow-infer p01-benchmark – run the full pipeline on the P01 sample

Fallback: if Typer is not installed, a minimal argparse entry point is used.
Both accept the same flags where practical.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

from ml.inference.config import InferenceConfig, load_config

logger = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _cfg_from_args(
    base: Optional[Path],
    backend: str,
    device: str,
    models: Optional[str],
    strategy: str,
) -> InferenceConfig:
    cfg = load_config(base)
    overrides = {}
    if backend:
        overrides["backend"] = backend
    if device:
        overrides["device"] = device
    if models:
        overrides["enabled_models"] = tuple(
            m.strip() for m in models.split(",") if m.strip()
        )
    if strategy:
        overrides["ensemble_strategy"] = strategy
    return cfg.with_(**overrides) if overrides else cfg


# ---------------------------------------------------------------------------
# Command implementations (Typer-independent)
# ---------------------------------------------------------------------------


def cmd_segment(
    input: Path,
    out: Path,
    *,
    backend: str = "",
    device: str = "",
    models: Optional[str] = None,
    strategy: str = "",
    config: Optional[Path] = None,
    verbose: bool = False,
    no_cache: bool = False,
) -> int:
    _configure_logging(verbose)
    from ml.inference.pipeline.segment import segment_study

    cfg = _cfg_from_args(config, backend, device, models, strategy)
    seg = segment_study(
        input,
        cfg,
        output_dir=out,
        use_cache=not no_cache,
    )
    print(json.dumps(seg.summary(), indent=2, default=str))
    return 0


def cmd_longitudinal(
    baseline: Path,
    followup: Path,
    out: Path,
    *,
    baseline_mask: Optional[Path] = None,
    followup_mask: Optional[Path] = None,
    date_a: Optional[str] = None,
    date_b: Optional[str] = None,
    backend: str = "",
    device: str = "",
    models: Optional[str] = None,
    strategy: str = "",
    config: Optional[Path] = None,
    verbose: bool = False,
    no_cache: bool = False,
) -> int:
    _configure_logging(verbose)
    from ml.inference.pipeline.longitudinal import compare_studies

    cfg = _cfg_from_args(config, backend, device, models, strategy)
    da = _parse_date(date_a)
    db = _parse_date(date_b)

    result = compare_studies(
        baseline,
        followup,
        cfg,
        date_a=da,
        date_b=db,
        output_dir=out,
        baseline_mask=baseline_mask,
        followup_mask=followup_mask,
        use_cache=not no_cache,
    )
    print(json.dumps(result.summary(), indent=2, default=str))
    return 0


def cmd_p01_benchmark(
    data: Path,
    out: Path,
    *,
    backend: str = "",
    device: str = "",
    models: Optional[str] = None,
    strategy: str = "",
    config: Optional[Path] = None,
    use_gt_masks: bool = False,
    verbose: bool = False,
) -> int:
    _configure_logging(verbose)
    from ml.inference.benchmark.p01 import run_p01_benchmark

    cfg = _cfg_from_args(config, backend, device, models, strategy)
    result = run_p01_benchmark(
        data_root=data,
        output_root=out,
        cfg=cfg,
        use_gt_masks=use_gt_masks,
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Unrecognised date format: {s!r}")


# ---------------------------------------------------------------------------
# Typer entry point (preferred)
# ---------------------------------------------------------------------------

try:
    import typer
except ImportError:  # pragma: no cover
    typer = None  # type: ignore[assignment]


def _build_typer_app():
    app = typer.Typer(  # type: ignore[union-attr]
        add_completion=False,
        help="OncoFlow inference CLI – 3-model panel, ensemble, longitudinal comparison.",
    )

    @app.command("segment")
    def segment(
        input: Path = typer.Option(..., exists=True, readable=True, help="Input NIfTI"),
        out: Path = typer.Option(..., help="Output directory"),
        backend: str = typer.Option("", help="local | gpu-prod"),
        device: str = typer.Option("", help="auto | cuda | mps | cpu"),
        models: Optional[str] = typer.Option(
            None, help="Comma list: nnunet,medgemma,sam3"
        ),
        strategy: str = typer.Option("", help="Ensemble strategy"),
        config: Optional[Path] = typer.Option(None, help="oncoflow.yaml"),
        verbose: bool = typer.Option(False),
        no_cache: bool = typer.Option(False, help="Skip cache reads"),
    ):
        sys.exit(cmd_segment(
            input, out,
            backend=backend, device=device,
            models=models, strategy=strategy,
            config=config, verbose=verbose, no_cache=no_cache,
        ))

    @app.command("longitudinal")
    def longitudinal(
        baseline: Path = typer.Option(..., exists=True),
        followup: Path = typer.Option(..., exists=True),
        out: Path = typer.Option(..., help="Output directory"),
        baseline_mask: Optional[Path] = typer.Option(None, exists=True),
        followup_mask: Optional[Path] = typer.Option(None, exists=True),
        date_a: Optional[str] = typer.Option(None, help="YYYY-MM-DD"),
        date_b: Optional[str] = typer.Option(None, help="YYYY-MM-DD"),
        backend: str = typer.Option(""),
        device: str = typer.Option(""),
        models: Optional[str] = typer.Option(None),
        strategy: str = typer.Option(""),
        config: Optional[Path] = typer.Option(None),
        verbose: bool = typer.Option(False),
        no_cache: bool = typer.Option(False),
    ):
        sys.exit(cmd_longitudinal(
            baseline, followup, out,
            baseline_mask=baseline_mask, followup_mask=followup_mask,
            date_a=date_a, date_b=date_b,
            backend=backend, device=device,
            models=models, strategy=strategy,
            config=config, verbose=verbose, no_cache=no_cache,
        ))

    @app.command("p01-benchmark")
    def p01_benchmark(
        data: Path = typer.Option(..., exists=True, dir_okay=True),
        out: Path = typer.Option(..., help="Output directory"),
        backend: str = typer.Option(""),
        device: str = typer.Option(""),
        models: Optional[str] = typer.Option(None),
        strategy: str = typer.Option(""),
        config: Optional[Path] = typer.Option(None),
        use_gt_masks: bool = typer.Option(
            False,
            help="Use ground-truth masks instead of the 3-model panel (sanity checks longitudinal pipeline only)",
        ),
        verbose: bool = typer.Option(False),
    ):
        sys.exit(cmd_p01_benchmark(
            data, out,
            backend=backend, device=device,
            models=models, strategy=strategy,
            config=config, use_gt_masks=use_gt_masks, verbose=verbose,
        ))

    return app


def main(argv: Optional[List[str]] = None) -> int:
    if typer is not None:
        app = _build_typer_app()
        app()
        return 0

    # Argparse fallback
    import argparse

    parser = argparse.ArgumentParser(prog="oncoflow-infer")
    sub = parser.add_subparsers(dest="cmd", required=True)

    seg = sub.add_parser("segment")
    seg.add_argument("--input", required=True, type=Path)
    seg.add_argument("--out", required=True, type=Path)
    seg.add_argument("--backend", default="")
    seg.add_argument("--device", default="")
    seg.add_argument("--models", default=None)
    seg.add_argument("--strategy", default="")
    seg.add_argument("--config", default=None, type=Path)
    seg.add_argument("--no-cache", action="store_true")
    seg.add_argument("--verbose", action="store_true")

    lon = sub.add_parser("longitudinal")
    lon.add_argument("--baseline", required=True, type=Path)
    lon.add_argument("--followup", required=True, type=Path)
    lon.add_argument("--out", required=True, type=Path)
    lon.add_argument("--baseline-mask", type=Path, default=None)
    lon.add_argument("--followup-mask", type=Path, default=None)
    lon.add_argument("--date-a", default=None)
    lon.add_argument("--date-b", default=None)
    lon.add_argument("--backend", default="")
    lon.add_argument("--device", default="")
    lon.add_argument("--models", default=None)
    lon.add_argument("--strategy", default="")
    lon.add_argument("--config", default=None, type=Path)
    lon.add_argument("--no-cache", action="store_true")
    lon.add_argument("--verbose", action="store_true")

    bench = sub.add_parser("p01-benchmark")
    bench.add_argument("--data", required=True, type=Path)
    bench.add_argument("--out", required=True, type=Path)
    bench.add_argument("--backend", default="")
    bench.add_argument("--device", default="")
    bench.add_argument("--models", default=None)
    bench.add_argument("--strategy", default="")
    bench.add_argument("--config", default=None, type=Path)
    bench.add_argument("--use-gt-masks", action="store_true")
    bench.add_argument("--verbose", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "segment":
        return cmd_segment(
            args.input, args.out,
            backend=args.backend, device=args.device,
            models=args.models, strategy=args.strategy,
            config=args.config, verbose=args.verbose,
            no_cache=args.no_cache,
        )
    if args.cmd == "longitudinal":
        return cmd_longitudinal(
            args.baseline, args.followup, args.out,
            baseline_mask=args.baseline_mask, followup_mask=args.followup_mask,
            date_a=args.date_a, date_b=args.date_b,
            backend=args.backend, device=args.device,
            models=args.models, strategy=args.strategy,
            config=args.config, verbose=args.verbose,
            no_cache=args.no_cache,
        )
    if args.cmd == "p01-benchmark":
        return cmd_p01_benchmark(
            args.data, args.out,
            backend=args.backend, device=args.device,
            models=args.models, strategy=args.strategy,
            config=args.config, use_gt_masks=args.use_gt_masks,
            verbose=args.verbose,
        )
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
