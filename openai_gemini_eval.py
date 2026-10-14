import os

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig


sample_data = [
    {
        "user_input": "What port does Ollama run on by default?",
        "retrieved_contexts": [
            "Ollama runs a local REST API server on port 11434 by default on localhost."
        ],
        "response": "By default, Ollama runs on port 11434.",
        "reference": "Ollama uses port 11434 by default.",
    }
]


def run_evaluation(provider, llm, embeddings):
    results = evaluate(
        dataset=EvaluationDataset.from_list(sample_data),
        metrics=[
            Faithfulness(),
            ResponseRelevancy(),
            LLMContextPrecisionWithReference(),
            LLMContextRecall(),
        ],
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        run_config=RunConfig(timeout=240, max_workers=1, max_retries=5),
    )
    print(f"\n{provider} results:")
    print(results.to_pandas())


def main():
    required_keys = ("OPENAI_API_KEY", "GOOGLE_API_KEY")
    missing_keys = [key for key in required_keys if not os.environ.get(key)]
    if missing_keys:
        raise SystemExit(
            "Set the following environment variables before running: "
            + ", ".join(missing_keys)
        )

    run_evaluation(
        "OpenAI",
        ChatOpenAI(model="gpt-4o-mini", temperature=0),
        OpenAIEmbeddings(model="text-embedding-3-small"),
    )
    run_evaluation(
        "Gemini",
        ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0),
        GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001"),
    )


if __name__ == "__main__":
    main()