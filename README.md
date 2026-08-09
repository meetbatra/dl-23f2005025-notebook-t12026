# DL & GenAI Project — Smart MCQ Solver

**Student:** Meet Batra  
**Roll Number:** 23f2005025

Smart MCQ Solver is a multiple-choice question answering system built for the DL & GenAI project. The repository contains the original notebook work, the trained BiLSTM inference service, the model artifacts required at runtime, and the supporting datasets and reports. The deployed FastAPI application takes a prompt plus five answer options, encodes each prompt-option pair, scores them with a BiLSTM model, and returns the top three candidates with softmax probabilities.

## What this repository contains

- `app/` — FastAPI application code for loading the model and serving predictions.
- `models/` — runtime model artifacts used by the API: `bilstm_model.pth` and `word2idx.json`.
- `datasets/` — local copies of `train.csv`, `test.csv`, and `sample_submission.csv` used by the notebooks.
- `notebooks/` — the project notebooks used for experimentation and milestone work.
- `reports/` — milestone report PDFs generated from the notebook analysis.
- `outputs/` — local inference outputs and saved model folders produced during testing.
- `requirements.txt` — pinned Python dependencies for the full project.
- `.python-version` — Python version hint for deployment/runtime alignment.

## Application architecture

The FastAPI service is split into three layers:

- `app/model.py` defines the `BiLSTMScorer` network and the preprocessing helpers.
- `app/inference.py` loads `bilstm_model.pth` and `word2idx.json` from `models/`, builds the input tensor, and produces predictions.
- `app/main.py` exposes the HTTP API and loads the model at startup.

The model path resolution is anchored to the repository root with `pathlib`, so the service works whether it is launched locally or from a deployment platform.

### BiLSTM model

The deployed classifier is a BiLSTM-based scorer configured as follows:

- embedding size: 100
- vocabulary size: 2,983
- hidden size: 128
- bidirectional LSTM
- dropout: 0.3
- output layer: linear projection to a single score per option

At inference time, each question is paired with each of the five options, encoded to a fixed length of 90 tokens, and scored independently. The five scores are passed through softmax and ranked to produce the top three answer choices.

### Preprocessing pipeline

The inference preprocessing is intentionally simple and matches the training-time pipeline used for the BiLSTM:

- `tokenize(text)` uses `re.findall(r'\w+', text.lower())`
- `combine_text(prompt, option)` builds `"{prompt} [SEP] {option}"`
- `encode(text, word2idx, max_len=90)` maps tokens to ids, uses index `1` for unknown tokens, truncates at 90 tokens, and pads with zeros
- `get_lengths(x)` counts non-zero tokens per sequence

## API endpoints

### `GET /`

Returns basic API metadata:

- API name
- short description
- available endpoints
- roll number

### `GET /health`

Returns the runtime status of the service:

- `status`: `ok`
- `model`: `BiLSTM`
- `vocab_size`: `2983`
- `max_len`: `90`

### `POST /predict`

Request body:

```json
{
  "prompt": "question text",
  "A": "option A",
  "B": "option B",
  "C": "option C",
  "D": "option D",
  "E": "option E"
}
```

Response:

```json
{
  "top3": ["B", "A", "C"],
  "scores": {
    "A": 0.12,
    "B": 0.89,
    "C": 0.08,
    "D": 0.03,
    "E": 0.01
  }
}
```

Scores are rounded to four decimal places. If inference fails, the API returns HTTP 500 with a descriptive message.

## Project files

- `app/__init__.py` — package marker for the FastAPI app.
- `app/main.py` — FastAPI entrypoint, request/response models, and endpoint definitions.
- `app/inference.py` — model loading, cached resource management, and prediction logic.
- `app/model.py` — BiLSTM model definition plus preprocessing helpers.
- `models/bilstm_model.pth` — trained BiLSTM weights loaded by the API.
- `models/word2idx.json` — vocabulary mapping used by the encoder.
- `models/README.md` — notes about the contents of the models folder.
- `notebooks/dl-23f2005025-notebook-t12026.ipynb` — main project notebook.
- `notebooks/milestone-3.ipynb` — milestone 3 notebook for retrieval-augmented context work.
- `notebooks/milestone-5.ipynb` — milestone 5 notebook for the later ensemble pipeline.
- `reports/milestone_3.pdf` — milestone 3 report.
- `reports/milestone_5.pdf` — milestone 5 report.
- `requirements.txt` — application and notebook dependencies.
- `.python-version` — runtime version hint.

## Local setup

Create and activate a virtual environment, then install the pinned dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the API locally:

```bash
uvicorn app.main:app --reload
```

For production-style execution, use the same app import path without `--reload`:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Deployment notes

The app is compatible with Render deployment as a Python web service. The important runtime details are:

- root directory: repository root
- build command: `pip install -r requirements.txt`
- start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- health check path: `/health`

The model files must remain in `models/` so the service can load them at startup. If they are missing, the app raises a clear startup error.

## Notebook dependencies

The notebooks use additional libraries beyond the API stack, including:

- `wandb`
- `sentence-transformers`
- `scikit-learn`
- `pandas`
- `matplotlib`
- `transformers`
- `datasets`
- `faiss-cpu`

These are included in `requirements.txt` so the notebooks remain reproducible locally.
