# ReClaim Documentation

Welcome to the ReClaim developer docs. This section evolves with the library—start here when planning API changes or onboarding contributors.

## Project goals
- Provide a clean foundation for automation and assistant workflows built on OpenAI + Pydantic.
- Encourage strong typing, validation, and reproducibility with uv.
- Make publishing to PyPI effortless.

## Getting started checklist
1. Install dependencies with `uv sync`.
2. Explore the package entry-point under `src/reclaim`.
3. Extend the README with feature-specific instructions as the library grows.
4. Keep usage snippets up to date by running them through `uv run python` before publishing.

## Release workflow
1. Update the version in `pyproject.toml` (and any user-facing references).
2. Run `uv lock --upgrade` if dependency constraints change.
3. Build artifacts via `uv build`.
4. Publish with `uv publish` or `twine upload dist/*`.

This document intentionally stays concise; treat it as the seed for more detailed guides, API references, or changelogs as ReClaim matures.
