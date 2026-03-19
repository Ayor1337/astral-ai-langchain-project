# Repository Guidelines

## Project Structure & Module Organization
`app/` contains the application code. Keep HTTP entrypoints in `app/api/`, configuration in `app/core/`, LLM integration in `app/llm/`, and request/response models in `app/schemas/`. `app/main.py` wires the FastAPI app and routers. Put automated tests in `tests/`; existing coverage focuses on config loading and the chat streaming API. Use `.env.example` as the template for local secrets, and `test_main.http` for manual endpoint checks in an HTTP client.

## Build, Test, and Development Commands
Create an isolated environment before installing dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the API locally with auto-reload:

```bash
uvicorn app.main:app --reload
```

Run the full test suite:

```bash
pytest
```

Run a focused test file when iterating on one area:

```bash
pytest tests/test_chat_api.py
```

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation and explicit imports. Use `snake_case` for functions, modules, and variables; use `PascalCase` for classes such as `Settings` and schema types. Keep route handlers thin and move provider-specific logic into `app/llm/`. Prefer small, pure helpers over defensive abstraction layers; this repository is intentionally simple.

## Testing Guidelines
Add tests beside the existing `tests/test_*.py` files. Name new tests `test_<behavior>()` and cover both success and failure paths, especially for streaming responses and configuration validation. Use `pytest` fixtures and `monkeypatch` for HTTP-layer tests; lightweight `unittest` cases are acceptable when they match the existing file. Before opening a PR, ensure `pytest` passes locally.

## Commit & Pull Request Guidelines
This workspace snapshot does not include `.git` metadata, so no repository-specific commit history is available. Use short, imperative commit messages such as `feat: add retry handling for upstream stream errors` or `test: cover invalid base URL`. Keep PRs focused, describe behavioral changes, list verification steps, and include sample requests or screenshots when API behavior or developer workflow changes.

## Security & Configuration Tips
Never commit real API keys in `.env`. Document any new environment variables in `.env.example` and validate them in `app/core/config.py`. If you add a new provider or external call, surface configuration errors early and return stable HTTP error responses.
