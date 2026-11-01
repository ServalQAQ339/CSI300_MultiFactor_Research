"""Tests for the four stage entrypoints and the ``python -m audits.<name>`` modules.

The 20 former root ``run_*.py`` scripts were removed after a static parity check against these
replacements (commit 0184b89); ``LEGACY_MAP`` keeps the historical mapping.
"""

import argparse
import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

SUBCOMMANDS = {
    "run_data": {"panels", "labels", "check-labels"},
    "run_factors": {"baseline", "backfill", "preprocess", "check-preprocess", "registry"},
    "run_research": {"ic", "diagnostics", "generate-candidates", "analyze-candidates", "select"},
    "run_backtest": {"models", "portfolio", "benchmark", "walk-forward"},
}

# Former script -> (new module, subcommand or None for a ``python -m audits.<name>`` module).
LEGACY_MAP = {
    "run_data_pipeline.py": ("run_data", "panels"),
    "run_label_pipeline.py": ("run_data", "labels"),
    "run_label_validation.py": ("run_data", "check-labels"),
    "run_factor_pipeline.py": ("run_factors", "baseline"),
    "run_factor_backfill.py": ("run_factors", "backfill"),
    "run_factor_preprocessing.py": ("run_factors", "preprocess"),
    "run_preprocessing_validation.py": ("run_factors", "check-preprocess"),
    "run_factor_registry.py": ("run_factors", "registry"),
    "run_factor_research.py": ("run_research", "ic"),
    "run_factor_diagnostics.py": ("run_research", "diagnostics"),
    "run_factor_expansion.py": ("run_research", ("generate-candidates", "analyze-candidates")),
    "run_factor_selection.py": ("run_research", "select"),
    "run_lightgbm.py": ("run_backtest", "models"),
    "run_portfolio_backtest.py": ("run_backtest", "portfolio"),
    "run_csi300_comparison.py": ("run_backtest", "benchmark"),
    "run_expanding_backtest.py": ("run_backtest", "walk-forward"),
    "run_factor_audit.py": ("audits.factor_audit", None),
    "run_enhanced_audit.py": ("audits.enhanced_audit", None),
    "run_factor_value_verification.py": ("audits.factor_value_verification", None),
    "run_factor_consistency_audit.py": ("audits.consistency_audit", None),
}

# Library function each replacement must reach, as the former script called it.
EXPECTED_CALLS = {
    ("run_data", "panels"): {"build_daily_dataset", "run_data_audit", "diagnose_missing_patterns"},
    ("run_data", "labels"): {"fetch_qlib_features", "build_labels", "summarize_price_audit"},
    ("run_data", "check-labels"): {"membership_diagnostics", "label_diagnostics"},
    ("run_factors", "baseline"): {"run_factor_engine"},
    ("run_factors", "backfill"): {"run_backfilled_factor_pipeline"},
    ("run_factors", "preprocess"): {"preprocess_factor_panel"},
    ("run_factors", "check-preprocess"): {"run_validation"},
    ("run_factors", "registry"): {"build_registry", "validate_registry", "write_registry"},
    ("run_research", "ic"): {"run"},
    ("run_research", "diagnostics"): {"ic_diagnostics", "quantile_diagnostics", "load_inputs"},
    ("run_research", "generate-candidates"): {"load_bars", "compute_factors", "build_registry"},
    ("run_research", "analyze-candidates"): {"load_research_panel", "nested_ridge_ic"},
    ("run_research", "select"): {"run_selection"},
    ("run_backtest", "models"): {"run_modeling"},
    ("run_backtest", "portfolio"): {"run_backtest"},
    ("run_backtest", "benchmark"): {"run_comparison"},
    ("run_backtest", "walk-forward"): {"run"},
    ("audits.factor_audit", None): {"run_factor_audit"},
    ("audits.enhanced_audit", None): {"run_enhanced_audit"},
    ("audits.factor_value_verification", None): {"run_audit"},
    ("audits.consistency_audit", None): {"audit_factor", "compare_saved", "prepare_panel"},
}
PROJECT_PACKAGES = {"config", "data", "factors", "labels", "factor_research", "modeling",
                    "portfolio", "audits"}


def _subparsers(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    """Return the subcommand name -> parser mapping of a stage parser."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return dict(action.choices)
    raise AssertionError("parser has no subcommands")


def _called_names(node: ast.AST) -> set[str]:
    """Return the plain function names called inside a node."""
    return {n.func.id for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}


def _reachable_calls(path: Path, start: str) -> set[str]:
    """Return names called from ``start``, following functions defined in the same module."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    seen, pending, calls = set(), [start], set()
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        found = _called_names(functions[name])
        calls |= found
        pending.extend(found)
    return calls


def _handler_name(module_name: str, command: str) -> str:
    """Return the handler function name a subcommand dispatches to."""
    parser = _subparsers(importlib.import_module(module_name).build_parser())[command]
    return parser.get_default("func").__name__


@pytest.mark.parametrize("module_name", sorted(SUBCOMMANDS))
def test_entrypoint_exposes_expected_subcommands(module_name):
    """Verify each stage entrypoint exposes exactly the planned subcommands with handlers.

    Description:
        Every subcommand must set ``func`` to a callable defined in the entrypoint module.
    Parameters:
        module_name (str): Root-level entrypoint module.
    Returns:
        None: Passes when the subcommand set and handlers match.
    Notes:
        Building the parser never runs a handler.
    """
    module = importlib.import_module(module_name)
    subparsers = _subparsers(module.build_parser())
    assert set(subparsers) == SUBCOMMANDS[module_name]
    for parser in subparsers.values():
        handler = parser.get_default("func")
        assert callable(handler) and handler.__module__ == module_name


@pytest.mark.parametrize(
    "module_name, command",
    [(m, c) for m, commands in sorted(SUBCOMMANDS.items()) for c in sorted(commands)],
)
def test_every_subcommand_parses_with_required_options(module_name, command):
    """Verify each subcommand parses when given placeholder values for required options.

    Description:
        Guards against argparse construction errors such as duplicate or malformed options.
    Parameters:
        module_name (str): Root-level entrypoint module.
        command (str): Subcommand name.
    Returns:
        None: Passes when parsing selects the subcommand handler.
    Notes:
        Only ``parse_args`` is called; the handler is not invoked.
    """
    parser = importlib.import_module(module_name).build_parser()
    required = [a.option_strings[0] for a in _subparsers(parser)[command]._actions
                if a.required and a.option_strings]
    argv = [command] + [token for option in required for token in (option, "x")]
    args = parser.parse_args(argv)
    assert args.command == command and callable(args.func)


def test_root_has_only_stage_entrypoints_and_mapping_is_complete():
    """Verify only the four stage entrypoints remain and every former script has a target.

    Description:
        The repository root holds exactly ``run_data.py``, ``run_factors.py``,
        ``run_research.py`` and ``run_backtest.py``; each of the 20 former scripts maps to an
        existing subcommand or audit ``main``.
    Parameters:
        None.
    Returns:
        None: Passes when the root is clean and the mapping resolves.
    Notes:
        ``LEGACY_MAP`` is the historical reference behind the runbook's mapping table.
    """
    assert {p.name for p in ROOT.glob("run_*.py")} == {f"{name}.py" for name in SUBCOMMANDS}
    assert len(LEGACY_MAP) == 20
    for module_name, command in LEGACY_MAP.values():
        commands = command if isinstance(command, tuple) else (command,)
        if command is None:
            assert callable(importlib.import_module(module_name).main)
        else:
            assert set(commands) <= SUBCOMMANDS[module_name]


@pytest.mark.parametrize("module_name, command", sorted(EXPECTED_CALLS, key=str))
def test_replacement_reaches_expected_library_functions(module_name, command):
    """Verify each subcommand or audit ``main`` still calls the library functions it wraps.

    Description:
        Follows calls from the handler through helpers defined in the same module, using
        ``ast`` only.
    Parameters:
        module_name (str): Entrypoint or audit module.
        command (str | None): Subcommand, or None for an audit module's ``main``.
    Returns:
        None: Passes when every expected function is reached.
    Notes:
        No handler is executed.
    """
    target = ROOT / (module_name.replace(".", "/") + ".py")
    start = "main" if command is None else _handler_name(module_name, command)
    missing = EXPECTED_CALLS[(module_name, command)] - _reachable_calls(target, start)
    assert not missing, f"{module_name} {command}: {missing}"


def test_expected_calls_cover_every_replacement():
    """Verify the call table covers every subcommand and audit module in the mapping."""
    mapped = set()
    for module_name, command in LEGACY_MAP.values():
        for cmd in command if isinstance(command, tuple) else (command,):
            mapped.add((module_name, cmd))
    assert mapped == set(EXPECTED_CALLS)


def test_benchmark_main_delegates_to_run_comparison():
    """Verify ``portfolio.benchmark.main`` and the new subcommand share one implementation."""
    assert "run_comparison" in _reachable_calls(ROOT / "portfolio" / "benchmark.py", "main")


def test_importing_entrypoints_has_no_side_effects(tmp_path):
    """Verify importing the stage entrypoints and audit modules runs nothing.

    Description:
        Imports all entrypoints in a fresh interpreter whose data root and Qlib directory
        point to non-existent temporary paths, then checks for output, created folders and
        heavy imports.
    Parameters:
        tmp_path (Path): Pytest temporary directory.
    Returns:
        None: Passes when nothing is printed or created and no heavy library is loaded.
    Notes:
        Audit modules import pandas at module level by design; only Qlib and LightGBM are
        forbidden there.
    """
    data_root, qlib_dir = tmp_path / "absent_data_root", tmp_path / "absent_qlib"
    code = (
        "import sys\n"
        "import run_data, run_factors, run_research, run_backtest\n"
        "heavy = sorted(m for m in ('pandas', 'numpy', 'qlib', 'lightgbm', 'sklearn')"
        " if m in sys.modules)\n"
        "import audits.factor_audit, audits.enhanced_audit\n"
        "import audits.factor_value_verification, audits.consistency_audit\n"
        "audit_heavy = sorted(m for m in ('qlib', 'lightgbm') if m in sys.modules)\n"
        "sys.stderr.write(repr((heavy, audit_heavy)))\n"
    )
    env = dict(os.environ, CSI300_DATA_ROOT=str(data_root), CSI300_QLIB_DIR=str(qlib_dir),
               PYTHONPATH=str(ROOT))
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert eval(result.stderr.strip().splitlines()[-1]) == ([], [])
    assert not data_root.exists() and not qlib_dir.exists()


def _module_path(module: str) -> Path | None:
    """Return the source file of a project module, or None when it does not exist."""
    base = ROOT.joinpath(*module.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.exists():
            return candidate
    return None


def _top_level_names(path: Path) -> set[str]:
    """Return names defined or imported at module level."""
    names = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


@pytest.mark.parametrize(
    "path", [f"{m}.py" for m in sorted(SUBCOMMANDS)]
    + [f"audits/{m}.py" for m in ("factor_audit", "enhanced_audit",
                                   "factor_value_verification", "consistency_audit")],
)
def test_lazy_imports_resolve(path):
    """Verify every project import in an entrypoint, including lazy ones, resolves statically.

    Description:
        Each ``from <project module> import name`` must point to an existing module that
        defines or re-exports ``name``.
    Parameters:
        path (str): Entrypoint file relative to the repository root.
    Returns:
        None: Passes when all imports resolve.
    Notes:
        Lazy imports are not executed by import-time tests, so they are checked with ``ast``.
    """
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.level == 0 and node.module
                and node.module.split(".")[0] in PROJECT_PACKAGES):
            source = _module_path(node.module)
            assert source is not None, f"{path}: {node.module}"
            missing = {a.name for a in node.names} - _top_level_names(source)
            assert not missing, f"{path}: {node.module} lacks {missing}"


def test_project_has_no_circular_top_level_imports():
    """Verify module-level imports between project modules form an acyclic graph.

    Description:
        Builds the graph of module-level absolute and relative project imports with ``ast``
        and searches it for cycles.
    Parameters:
        None.
    Returns:
        None: Passes when no cycle exists.
    Notes:
        Function-level (lazy) imports cannot create import-time cycles and are ignored.
    """
    skipped = {"archive", "exploratory", "tests", ".git", ".venv", "venv"}
    files = [p for p in ROOT.rglob("*.py") if not skipped & set(p.relative_to(ROOT).parts)]
    graph = {}
    for path in files:
        parts = path.relative_to(ROOT).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        package = module.split(".")[:-1] if parts[-1] != "__init__" else module.split(".")
        edges = set()
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    base = package[: len(package) - node.level + 1]
                    target = ".".join(base + ([node.module] if node.module else []))
                else:
                    target = node.module or ""
            elif isinstance(node, ast.Import):
                target = node.names[0].name
            else:
                continue
            if target.split(".")[0] in PROJECT_PACKAGES and _module_path(target):
                edges.add(target)
        graph[module] = edges

    state = {}

    def visit(node, stack):
        state[node] = "active"
        for nxt in graph.get(node, ()):
            if state.get(nxt) == "active":
                raise AssertionError(f"import cycle: {' -> '.join(stack + [nxt])}")
            if nxt not in state:
                visit(nxt, stack + [nxt])
        state[node] = "done"

    for module in graph:
        if module not in state:
            visit(module, [module])
