<!--
SYNC IMPACT REPORT
==================
Version change: [UNVERSIONED TEMPLATE] → 1.0.0
Bump rationale: MINOR — Initial population of all principle content from blank template;
                all five principles defined, two additional sections added.

Modified principles:
  [PRINCIPLE_1_NAME] → I. Code Quality (NON-NEGOTIABLE)
  [PRINCIPLE_2_NAME] → II. Test-Driven Development (NON-NEGOTIABLE)
  [PRINCIPLE_3_NAME] → III. Review Standards
  [PRINCIPLE_4_NAME] → IV. Consistency
  [PRINCIPLE_5_NAME] → V. Performance

Added sections:
  - Quality Gates (automated enforcement layer)
  - Development Workflow

Removed sections: none

Templates requiring updates:
  ✅ .specify/templates/plan-template.md — "Constitution Check" section already generic;
     gates now map to the five principles defined here.
  ✅ .specify/templates/spec-template.md — no structural changes required; Success Criteria
     section already supports measurable performance metrics (SC-002 pattern).
  ✅ .specify/templates/tasks-template.md — Phase N (Polish) already includes security
     hardening, performance optimization, and linting steps that align with this constitution.
  ⚠ .specify/templates/commands/ — no command files found; no updates required.

Deferred TODOs: none — all fields resolved from repo context.
-->

# pcg-llm Constitution

## Core Principles

### I. Code Quality (NON-NEGOTIABLE)

All production code in `src/` MUST pass every automated quality gate before merging:

- **Linting & formatting**: Ruff MUST report zero errors and zero warnings. Auto-fixes are
  permitted, but the fixed result MUST still be committed and pass cleanly.
- **Type safety**: Mypy MUST pass with `disallow_untyped_defs = true` and no suppressed
  errors except `# type: ignore` lines that carry an explicit inline justification comment.
- **Security scanning**: Bandit MUST report no issues of severity MEDIUM or HIGH.
  LOW-severity findings MAY be suppressed only with a `# nosec` comment and an inline
  explanation of why the risk is accepted.
- Pre-commit hooks enforce all of the above locally; CI enforces them globally. A PR
  MUST NOT be merged if CI is red.

**Rationale**: Automated gates catch entire categories of defect (type errors, style drift,
common vulnerabilities) at zero marginal review cost. Manual review bandwidth is reserved
for logic and architecture, not style.

### II. Test-Driven Development (NON-NEGOTIABLE)

Tests MUST be written before implementation code for every new feature or bug-fix:

- **Red-Green-Refactor**: Write a failing test → confirm it fails → implement → confirm
  it passes → refactor. Skipping any step is a violation.
- **Coverage floor**: The pytest `--cov-fail-under=80` threshold is the minimum; new
  modules SHOULD target ≥90 % branch coverage. Coverage MUST NOT decrease across a PR.
- **Test categories** (use pytest markers): `unit` for isolated logic, `integration` for
  multi-component flows, `convergence` for DEQ solver verification, `gpu` for
  hardware-dependent paths, `slow` for anything >5 s on CPU.
- Integration and convergence tests MUST be included whenever a change touches the DEQ
  solver, the routing graph, or inter-module contracts.
- Tests in `tests/` MUST follow the `test_*.py` / `Test*` / `test_*` naming conventions
  enforced by `name-tests-test` pre-commit hook.

**Rationale**: PCG-LLM's correctness depends on convergence properties that are invisible
without explicit tests. Retrofitting tests after implementation consistently misses
edge-case behaviour in iterative/dynamical systems.

### III. Review Standards

Every change that modifies `src/` or `tests/` MUST go through a pull request with at
least one approving review before merging to `main`:

- The PR author MUST run `pre-commit run --all-files` and fix all findings locally before
  opening the PR — CI failure caused by a hook violation is not a reviewer's problem.
- The PR description MUST reference the spec or ADR that motivates the change, or include
  an inline rationale if no spec exists.
- Reviewers MUST verify: (a) tests exist and were written before implementation (TDD),
  (b) type annotations are complete, (c) no new `# type: ignore` or `# nosec` without
  justification, (d) the change does not decrease coverage.
- Draft PRs are permitted for early feedback but MUST NOT be merged until all CI checks
  are green and review is approved.
- Emergency hotfixes to `main` require at minimum a post-merge retrospective item
  documenting what bypass occurred and why.

**Rationale**: Peer review is the last human gate before code reaches the shared codebase.
Consistent standards prevent "LGTM" rubber-stamping and keep the bar uniform across
contributors.

### IV. Consistency

All code in the repository MUST conform to a single coherent style and structure:

- **Python version**: Python 3.13 exclusively. No compatibility shims for older versions.
- **Line length**: 100 characters (Ruff enforced). No per-file overrides without
  documented justification in `pyproject.toml`.
- **Imports**: `isort` rules enforced via Ruff `I` rule set. Absolute imports MUST be
  used throughout `src/`; relative imports are permitted only inside a single package.
- **Naming**: PEP 8 enforced via Ruff `N` rule set. Public API names MUST be stable
  across minor versions; rename via deprecation alias when breaking.
- **Source layout**: `src/pcg_llm/` layout strictly maintained. No top-level `*.py`
  modules except `main.py` (entry point) and `conftest.py` (pytest root fixture).
- **File hygiene**: No trailing whitespace, LF line endings, and a single newline at
  end-of-file (enforced by pre-commit hooks).

**Rationale**: Consistent style eliminates trivial diff noise, accelerates code review,
and prevents subtle bugs (e.g., import ordering affecting circular-dependency detection).

### V. Performance

Performance characteristics MUST be measured and MUST NOT silently regress:

- Any change to the DEQ solver, routing graph, or attention mechanism MUST include or
  update a benchmark in `tests/` (marked `@pytest.mark.slow` or a dedicated
  `benchmarks/` suite) that captures wall-clock time and memory usage on a reference
  input size.
- Performance targets are feature-specific and MUST be documented in the relevant spec.
  In the absence of a spec target, the baseline is the measured value on `main` at the
  time of the PR; the PR MUST NOT introduce a regression >5 % on that baseline.
- GPU-dependent performance MUST be measured under `@pytest.mark.gpu` and is exempt from
  CPU-only CI runs, but MUST be verified before merging performance-sensitive changes.
- YAGNI applies to optimisation: do not pre-optimise code paths that are not yet
  measured bottlenecks. Profile first, then optimise.

**Rationale**: PCG-LLM is a research architecture where convergence speed and memory
footprint are first-class concerns. Silent regressions compound across experiments and
invalidate comparative results.

## Quality Gates

These gates are the automated enforcement layer for the principles above. All gates MUST
pass before a branch is eligible for merge.

| Gate | Tool | Trigger | Failure action |
|------|------|---------|----------------|
| Linting | `ruff check` | pre-commit + CI | Fix before push |
| Formatting | `ruff format` | pre-commit + CI | Auto-fix, re-commit |
| Type checking | `mypy src/` | pre-commit + CI | Fix annotations |
| Security scan | `bandit -r src/` | CI | Fix or document nosec |
| Test suite | `pytest` | CI | Fix failing tests |
| Coverage floor | `--cov-fail-under=80` | CI | Add tests |
| YAML/TOML/JSON validity | pre-commit hooks | pre-commit + CI | Fix syntax |
| Merge conflict markers | pre-commit hook | pre-commit | Resolve conflicts |
| Private key detection | pre-commit hook | pre-commit | Remove secrets |

The CI pipeline (`.github/workflows/ci.yml`) is the authoritative gate; passing locally
is necessary but not sufficient.

## Development Workflow

The standard flow for all feature work is:

1. **Spec first**: Open or reference a spec in `specs/` before writing code. ADRs MUST
   be written for architectural decisions that affect the DEQ solver, graph topology, or
   public API surface.
2. **Branch**: Create a feature branch from `main`. Branch names MUST follow the pattern
   `###-short-description` (e.g., `007-deq-convergence-tests`).
3. **TDD cycle**: Write failing tests → implement → pass → refactor (Principle II).
4. **Local gates**: Run `pre-commit run --all-files` and `pytest` before pushing.
5. **PR**: Open against `main`. Title MUST summarise the change; body MUST link to spec
   or include rationale. All CI checks MUST be green before requesting review.
6. **Review**: Reviewer checks against Principle III checklist. At least one approval
   required.
7. **Merge**: Squash or merge commit — project convention is squash for single-concern
   PRs, merge commit for multi-commit feature branches. Delete branch after merge.
8. **Changelog**: Update `CHANGELOG.md` under `[Unreleased]` for every user-visible
   change (features, fixes, breaking changes, deprecations).

## Governance

- This constitution supersedes all other practices, conventions, and tool defaults in
  the repository. When a conflict exists, the constitution wins.
- **Amendments** require: (a) a written rationale, (b) a version bump per the policy
  below, (c) updating this file and the Sync Impact Report comment, and (d) propagating
  changes to affected templates per the checklist in the `speckit-constitution` command.
- **Versioning policy**:
  - MAJOR: Backward-incompatible governance changes — removing or fundamentally
    redefining a principle.
  - MINOR: Adding a new principle, section, or materially expanding guidance.
  - PATCH: Clarifications, wording improvements, typo fixes.
- **Compliance review**: Every sprint retrospective SHOULD include a one-item check —
  "Did any merged PR violate the constitution?" If yes, file a follow-up task to
  address the root cause and amend the constitution if the violation reveals a gap.
- **Runtime development guidance**: Refer to `.claude/` agent files for AI-assisted
  development workflows within this project.

**Version**: 1.0.0 | **Ratified**: 2026-04-04 | **Last Amended**: 2026-04-04
