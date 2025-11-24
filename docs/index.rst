ReClaim Documentation
=====================

Welcome to the ReClaim docs. These pages target contributors who need to understand how the library decomposes text into claims, aligns them to model tokens, and annotates them for faithfulness and factuality.

.. toctree::
   :maxdepth: 2

   api_reference

Getting Started
---------------

- Install dependencies with ``uv sync``.
- Explore the package entry point under ``src/reclaim``.
- Run the usage snippets from ``README.md`` to validate your environment.

Release Workflow
----------------

1. Update the version in ``pyproject.toml`` (and any user-facing references).
2. Run ``uv lock --upgrade`` if dependency constraints change.
3. Build artifacts via ``uv build``.
4. Publish with ``uv publish`` or ``twine upload dist/*``.
