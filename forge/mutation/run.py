#!/usr/bin/env python3
"""Runs Foundry mutation campaigns per target and enforces the mutation policy.

`forge test --mutate` reports surviving mutants but still exits 0, and its score
excludes invalid, skipped and timed-out mutants, so the gate has to live outside
Forge. This module runs one campaign per target defined in `targets.toml`, parses
the `--json` report, and exits non-zero when a target regresses.

Targets are evaluated individually rather than against a repository-wide average,
so a strong score on one subsystem cannot mask survivors in another.

Typical usage example:

    # PR gate: critical targets plus whatever the branch touched
    python3 mutation/run.py --changed-since origin/main

    # Nightly: everything, report-only
    python3 mutation/run.py --all --report-only

    # One subsystem, locally
    python3 mutation/run.py --target access
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path

FORGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = FORGE_ROOT / "mutation" / "targets.toml"


@dataclass
class Target:
    """A group of contracts mutated and gated together.

    Attributes:
        name: The target's key in `targets.toml`, used on the command line.
        paths: Production contracts to mutate, relative to the forge directory.
        min_score: Score floor, below which the campaign fails. Taken from an
          observed minimum less a margin, never from a single measurement.
        critical: Whether to run on every pull request even when untouched.
        max_invalid_rate: Ceiling on the share of mutants that fail to compile,
          overriding the global default when set.
    """

    name: str
    paths: list[str]
    min_score: float
    critical: bool = False
    max_invalid_rate: float | None = None


@dataclass
class Defaults:
    """Campaign settings shared by every target.

    Attributes:
        jobs: Mutation workers to run in parallel. Pinned rather than left to
          the core count, which would make results runner-dependent.
        timeout: Per-mutant wall clock in seconds.
        max_invalid_rate: Ceiling on the share of mutants that fail to compile,
          for targets that do not set their own.
    """

    jobs: int = 4
    timeout: int = 30
    max_invalid_rate: float = 50.0


@dataclass
class Result:
    """The outcome of one target's mutation campaign.

    Attributes:
        target: Name of the target this result belongs to.
        summary: Forge's `summary` object, or empty if the campaign never ran.
        survivors: Surviving mutants keyed by source path, as Forge reports them.
        failures: Policy violations found; empty means the target passed.
    """

    target: str
    summary: dict
    survivors: dict
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the target satisfied every policy check."""
        return not self.failures


def load_config(path: Path) -> tuple[Defaults, dict[str, Target]]:
    """Reads the mutation policy from a TOML file.

    Unknown keys are rejected rather than ignored, in both `[defaults]` and each
    `[targets.<name>]`: a misspelled key would otherwise leave a policy field at
    its permissive default and silently weaken the gate.

    Args:
        path: Location of the policy file, normally `mutation/targets.toml`.

    Returns:
        A tuple `(defaults, targets)`, where `defaults` holds the shared campaign
        settings and `targets` maps each target name to its definition.

    Raises:
        SystemExit: If the file defines no targets or contains an unknown key.
    """
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    known = {f.name for f in fields(Defaults)}
    unknown = sorted(raw.get("defaults", {}).keys() - known)
    if unknown:
        sys.exit(
            f"unknown key(s) in [defaults] of {path}: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(known))})"
        )
    defaults = Defaults(**raw.get("defaults", {}))

    target_keys = {f.name for f in fields(Target)} - {"name"}
    targets: dict[str, Target] = {}
    for name, body in raw.get("targets", {}).items():
        # Without this, a typo like `min_socre` would leave min_score at 0.0 and
        # silently disable the floor for that target.
        unknown = sorted(body.keys() - target_keys)
        if unknown:
            sys.exit(
                f"unknown key(s) in [targets.{name}] of {path}: "
                f"{', '.join(unknown)} (known: {', '.join(sorted(target_keys))})"
            )
        targets[name] = Target(
            name=name,
            paths=list(body["paths"]),
            min_score=float(body.get("min_score", 0.0)),
            critical=bool(body.get("critical", False)),
            max_invalid_rate=(
                float(body["max_invalid_rate"]) if "max_invalid_rate" in body else None
            ),
        )
    if not targets:
        sys.exit(f"no targets defined in {path}")
    return defaults, targets


def validate_paths(targets: dict[str, Target]) -> None:
    """A silently missing path would shrink the campaign without failing it."""
    missing = [
        f"{target.name}: {p}"
        for target in targets.values()
        for p in target.paths
        if not (FORGE_ROOT / p).is_file()
    ]
    if missing:
        sys.exit("configured mutation paths do not exist:\n  " + "\n  ".join(missing))


def changed_sources(base_ref: str) -> list[str]:
    """Finds the Solidity files this branch changed.

    Test files count, and so do deletions: weakening or removing
    `test/swarm/SwarmCore.t.sol` lowers the swarm score just as surely as editing
    the contract does, and a regression gate that ignores that is trivially
    bypassed.

    Args:
        base_ref: Branch or revision to compare against, such as `origin/main`.
          The comparison runs from its merge base with HEAD where one exists.

    Returns:
        Paths of the changed `.sol` files under `src` or `test`, relative to the
        forge directory. Empty when the branch changed no Solidity.

    Raises:
        SystemExit: If the diff against `base_ref` fails.
    """
    merge_base = subprocess.run(
        ["git", "merge-base", base_ref, "HEAD"],
        cwd=FORGE_ROOT,
        capture_output=True,
        text=True,
    )
    diff_from = merge_base.stdout.strip() if merge_base.returncode == 0 else base_ref

    # Deletions are deliberately included: removing a test is the strongest way
    # to weaken a subsystem, so it has to select that subsystem's target. Only
    # the path string is used here, never the file, and a deleted *target* path
    # is caught separately by validate_paths.
    diff = subprocess.run(
        ["git", "diff", "--name-only", diff_from, "HEAD"],
        cwd=FORGE_ROOT,
        capture_output=True,
        text=True,
    )
    if diff.returncode != 0:
        sys.exit(f"git diff against {base_ref} failed: {diff.stderr.strip()}")

    return [
        line[len("forge/") :]
        for line in diff.stdout.splitlines()
        if line.endswith(".sol")
        and (line.startswith("forge/src/") or line.startswith("forge/test/"))
    ]


def subsystem_of(path: str) -> str | None:
    """Returns the subsystem directory owning a path, or None if it has none.

    `src/swarm/registry/SwarmCore.sol` and `test/swarm/X.t.sol` both give
    `swarm`: `src` and `test` mirror each other one directory deep, which is what
    lets a changed test select the target that owns the contracts it covers.
    A path shallower than that, such as a helper directly under `test`, belongs
    to no single subsystem and yields None.

    Args:
        path: A repository path relative to the forge directory.
    """
    parts = Path(path).parts
    return parts[1] if len(parts) > 2 else None


def select_targets(
    args: argparse.Namespace, targets: dict[str, Target]
) -> list[Target]:
    """Decides which targets to run for this invocation.

    Under `--changed-since`, selection is the union of the targets marked
    critical and those owning a changed subsystem. A changed subsystem that no
    target owns is reported as a warning rather than passing unnoticed.

    Args:
        args: Parsed arguments carrying exactly one of `all`, `target` or
          `changed_since`.
        targets: Every target defined by the policy, keyed by name.

    Returns:
        The targets to run, in policy order for `--all` and alphabetical order
        for `--changed-since`. May be empty when nothing needs mutating.

    Raises:
        SystemExit: If no selection mode was given, or a named target is unknown.
    """
    if args.all:
        return list(targets.values())

    if args.target:
        unknown = sorted(set(args.target) - targets.keys())
        if unknown:
            sys.exit(
                f"unknown target(s): {', '.join(unknown)} "
                f"(known: {', '.join(sorted(targets))})"
            )
        return [targets[name] for name in args.target]

    if args.changed_since:
        touched = sorted(set(changed_sources(args.changed_since)))
        if touched:
            print(f"changed Solidity sources: {', '.join(touched)}")
        else:
            print("no Solidity sources changed")

        subsystems = {subsystem_of(path) for path in touched} - {None}
        selected = sorted(
            target.name
            for target in targets.values()
            if target.critical
            or subsystems.intersection(subsystem_of(p) for p in target.paths)
        )

        unowned = subsystems - {
            subsystem_of(p) for t in targets.values() for p in t.paths
        }
        if unowned:
            print(
                "warning: no mutation target owns "
                f"{', '.join(sorted(unowned))} — add one to targets.toml"
            )
        return [targets[name] for name in selected]

    sys.exit("one of --all, --target or --changed-since is required")


def build_command(target: Target, defaults: Defaults, extra: list[str]) -> list[str]:
    """Builds the `forge test --mutate` argument vector for one target.

    Args:
        target: The target whose paths are to be mutated.
        defaults: Campaign settings supplying worker count and per-mutant timeout.
        extra: Additional arguments forwarded verbatim to `forge test`.

    Returns:
        The command to execute, suitable for `subprocess.run`.
    """
    return [
        "forge",
        "test",
        "--mutate",
        *target.paths,
        "--mutation-jobs",
        str(defaults.jobs),
        "--mutation-timeout",
        str(defaults.timeout),
        *extra,
        "--json",
    ]


def run_campaign(target: Target, defaults: Defaults, extra: list[str]) -> Result:
    """Runs one target's mutation campaign and evaluates it against the policy.

    A forge failure or an unparseable report is recorded as a policy violation
    rather than raised, so that remaining targets still run and the summary
    reports every problem at once.

    Args:
        target: The target to mutate.
        defaults: Campaign settings for this run.
        extra: Additional arguments forwarded verbatim to `forge test`.

    Returns:
        The campaign result, carrying any violations found.
    """
    command = build_command(target, defaults, extra)
    print(f"\n::group::mutation campaign: {target.name}")
    print("$ " + " ".join(command), flush=True)

    completed = subprocess.run(command, cwd=FORGE_ROOT, capture_output=True, text=True)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    print("::endgroup::", flush=True)

    if completed.returncode != 0:
        return Result(
            target=target.name,
            summary={},
            survivors={},
            failures=[f"forge exited {completed.returncode}"],
        )

    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return Result(
            target=target.name,
            summary={},
            survivors={},
            failures=[f"could not parse the forge JSON report: {exc}"],
        )

    return evaluate(target, defaults, report)


def evaluate(target: Target, defaults: Defaults, report: dict) -> Result:
    """Applies the mutation policy to one Forge report.

    A target fails if no mutant was actually evaluated, if any mutant timed out,
    if too many mutants were invalid, or if the score is below the target's
    floor. The first two matter because Forge excludes both from its score: a
    campaign producing only invalid mutants would otherwise report a perfect
    result over an empty denominator, and a timeout would silently leave the
    denominator rather than count as an escape.

    Args:
        target: The target the report belongs to, supplying its floor.
        defaults: Campaign settings, used for the fallback invalid-rate ceiling.
        report: The parsed `--json` output of `forge test --mutate`.

    Returns:
        The result, whose `failures` list is empty when the target passed.
    """
    summary = report.get("summary", {})
    survivors = report.get("survived_mutants", {})

    killed = summary.get("killed", 0)
    survived = summary.get("survived", 0)
    invalid = summary.get("invalid", 0)
    timed_out = summary.get("timed_out", 0)
    total = summary.get("total", 0)
    score = summary.get("mutation_score", 0.0)

    failures: list[str] = []

    # A setup that generates only invalid mutants would otherwise report a
    # perfect score over an empty denominator and pass silently.
    evaluated = killed + survived
    if evaluated == 0:
        failures.append(
            f"no mutant was evaluated ({total} generated, {invalid} invalid, "
            f"{summary.get('skipped', 0)} skipped) — the campaign proves nothing"
        )

    # Forge excludes timeouts from the score, so an unnoticed timeout silently
    # removes a mutant from the denominator rather than counting as a kill.
    if timed_out:
        failures.append(
            f"{timed_out} mutant(s) timed out after {defaults.timeout}s and were "
            "excluded from the score — investigate before trusting this run"
        )

    ceiling = (
        target.max_invalid_rate
        if target.max_invalid_rate is not None
        else defaults.max_invalid_rate
    )
    if total:
        invalid_rate = 100.0 * invalid / total
        if invalid_rate > ceiling:
            failures.append(
                f"invalid-mutant rate {invalid_rate:.1f}% exceeds the "
                f"{ceiling:.1f}% ceiling ({invalid}/{total}) — the mutation "
                "engine and the contract have probably drifted apart"
            )

    if evaluated and score < target.min_score:
        failures.append(
            f"mutation score {score:.2f}% is below the {target.min_score:.2f}% "
            f"floor for '{target.name}' ({killed} killed / {survived} survived)"
        )

    return Result(target.name, summary, survivors, failures)


def format_summary(results: list[Result], report_only: bool) -> str:
    """Renders campaign results as a GitHub job summary.

    Args:
        results: One result per target that ran, in the order they ran.
        report_only: Whether violations are being reported without failing, which
          changes how each target's gate column reads.

    Returns:
        A Markdown report: a per-target table, a collapsed list of every
        surviving mutant with its source rewrite, and any policy violations.
    """
    lines = ["## Mutation testing", ""]
    lines.append(
        "| Target | Score | Killed | Survived | Invalid | Skipped | Timed out | Gate |"
    )
    lines.append("| --- | --: | --: | --: | --: | --: | --: | :-: |")
    for result in results:
        s = result.summary
        gate = "report-only" if report_only else ("pass" if result.ok else "**fail**")
        lines.append(
            f"| `{result.target}` "
            f"| {s.get('mutation_score', 0.0):.2f}% "
            f"| {s.get('killed', 0)} "
            f"| {s.get('survived', 0)} "
            f"| {s.get('invalid', 0)} "
            f"| {s.get('skipped', 0)} "
            f"| {s.get('timed_out', 0)} "
            f"| {gate} |"
        )

    survivor_lines: list[str] = []
    for result in results:
        for path, mutants in sorted(result.survivors.items()):
            for mutant in mutants:
                survivor_lines.append(
                    f"- `{path}:{mutant['line']}` — "
                    f"`{mutant['original']}` → `{mutant['mutant']}`"
                )
    if survivor_lines:
        lines += [
            "",
            f"<details><summary>{len(survivor_lines)} surviving mutant(s)</summary>",
            "",
            *survivor_lines,
            "",
            "Each survivor needs a behavioural assertion, a fuzz boundary case, an",
            "invariant, or a documented equivalent-mutant waiver.",
            "",
            "</details>",
        ]

    problems = [f"- `{r.target}`: {f}" for r in results for f in r.failures]
    if problems:
        lines += ["", "### Policy violations", "", *problems]

    return "\n".join(lines) + "\n"


def main() -> int:
    """Runs the selected mutation campaigns and applies the policy.

    Returns:
        A process exit status: 0 when every selected target satisfied the policy
        or `--report-only` was given, and 1 when any target violated it.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all", action="store_true", help="run every target")
    selection.add_argument(
        "--target", action="append", default=[], help="run a named target (repeatable)"
    )
    selection.add_argument(
        "--changed-since",
        metavar="REF",
        help="run critical targets plus any target touched since REF",
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--jobs", type=int, help="override the configured mutation worker count"
    )
    parser.add_argument(
        "--timeout", type=int, help="override the configured per-mutant timeout"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the campaigns that would run, then exit",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="report violations but always exit 0",
    )
    parser.add_argument(
        "--report", type=Path, help="write the aggregated JSON report to this path"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=os.environ.get("GITHUB_STEP_SUMMARY"),
        help="append a Markdown summary to this path",
    )
    parser.add_argument(
        "--forge-arg",
        action="append",
        default=[],
        help="extra argument forwarded to `forge test` (repeatable)",
    )
    args = parser.parse_args()

    defaults, targets = load_config(args.config)
    if args.jobs is not None:
        defaults.jobs = args.jobs
    if args.timeout is not None:
        defaults.timeout = args.timeout
    validate_paths(targets)
    selected = select_targets(args, targets)

    if not selected:
        print("no mutation target selected — nothing to do")
        return 0

    print(f"targets: {', '.join(t.name for t in selected)}")

    if args.dry_run:
        for target in selected:
            print("$ " + " ".join(build_command(target, defaults, args.forge_arg)))
        return 0

    results = [run_campaign(target, defaults, args.forge_arg) for target in selected]

    report = {
        "report_only": args.report_only,
        "defaults": vars(defaults),
        "targets": {
            r.target: {
                "min_score": targets[r.target].min_score,
                "critical": targets[r.target].critical,
                "summary": r.summary,
                "survived_mutants": r.survivors,
                "violations": r.failures,
            }
            for r in results
        },
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")

    rendered = format_summary(results, args.report_only)
    print("\n" + rendered)
    if args.summary:
        with open(args.summary, "a", encoding="utf-8") as handle:
            handle.write(rendered)

    failed = [r for r in results if not r.ok]
    if not failed:
        print("mutation policy satisfied")
        return 0
    if args.report_only:
        print(f"{len(failed)} target(s) violate the policy (report-only, not failing)")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
