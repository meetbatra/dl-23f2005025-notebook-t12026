# FastAPI entrypoint exposing the Smart MCQ Solver prediction API.

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .inference import load_resources, predict
from .model import MAX_LEN, VOCAB_SIZE


class PredictionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    A: str = Field(..., min_length=1)
    B: str = Field(..., min_length=1)
    C: str = Field(..., min_length=1)
    D: str = Field(..., min_length=1)
    E: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    top3: list[str]
    scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model: str
    vocab_size: int
    max_len: int


class RootResponse(BaseModel):
    api_name: str
    description: str
    available_endpoints: list[str]
    roll_number: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        load_resources()
    except Exception as exc:
        raise RuntimeError(str(exc)) from exc
    yield


app = FastAPI(
    title="Smart MCQ Solver API",
    description="BiLSTM-based multiple choice question solver.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return RootResponse(
        api_name="Smart MCQ Solver API",
        description=(
            "Smart MCQ Solver — predicts the most likely answer from 5 options "
            "using a BiLSTM model trained from scratch"
        ),
        available_endpoints=[
            "POST /predict",
            "GET /health",
            "GET /",
        ],
        roll_number="23f2005025",
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="BiLSTM",
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(payload: PredictionRequest) -> PredictionResponse:
    try:
        result = predict(
            prompt=payload.prompt,
            a=payload.A,
            b=payload.B,
            c=payload.C,
            d=payload.D,
            e=payload.E,
        )
        return PredictionResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
