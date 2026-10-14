# Ragas Qwen Evaluator (Evaluation framework for RAG and LLM applications)
A small Ragas evaluation script that uses Qwen models served locally by Ollama. It evaluates a sample answer for faithfulness, response relevancy, and context precision and recall.

## Key Features

- Objective metrics for evaluating LLM applications with LLM-based and traditional metrics.
- Test data generation for creating comprehensive scenarios.
- Integrations with popular LLM frameworks and observability tools.
- Feedback loops to improve applications using production data.

## Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (uv will manage the Python environment and dependencies)
- [Ollama](https://ollama.com/download) installed and running
- Enough local memory and disk space to run and store the models below

## Installation

### 1. Install uv

On Windows, install uv with WinGet:

```powershell
winget install --id=astral-sh.uv -e
```

Open a new PowerShell terminal after installation, then navigate to the project directory. Sync the project dependencies:

```powershell
uv sync
```

`uv sync` creates the project's `.venv` and installs the dependencies declared in `pyproject.toml`. The project requires Python 3.12 or newer; uv can download a compatible Python version if needed.

### 2. Download the Ollama models

Make sure Ollama is running, then download both models used by the script:

```powershell
ollama pull qwen3.5:4b
ollama pull qwen3-embedding:0.6b
```

Ollama normally serves its local API at `http://localhost:11434`, which is the default endpoint used by the script.

## Run

With Ollama running, start the script through uv:

```powershell
uv run main.py
```

The script evaluates its built-in sample dataset and prints the results as a table. The first run may take longer while the models load.

## OpenAI and Gemini evaluation

The `openai_gemini_eval.py` script evaluates the same sample with OpenAI and Gemini. Set API keys for both providers in PowerShell, then run it:

```powershell
$env:OPENAI_API_KEY = "your-openai-api-key"
$env:GOOGLE_API_KEY = "your-google-api-key"
uv run openai_gemini_eval.py
```

The script uses OpenAI `gpt-4o-mini` with `text-embedding-3-small`, and Gemini `gemini-2.5-flash` with `gemini-embedding-001`. Both API keys are required because both evaluations run in sequence.
