from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)

# 1. Configure qwen3.5:4b (CRITICAL: disable reasoning so Ragas gets clean JSON)
ollama_llm = ChatOllama(
    model="qwen3.5:4b",
    temperature=0,  # Keep deterministic for consistent grading
    reasoning=False,  # Disables <think>...</think> blocks that break Ragas JSON parsing
    num_ctx=8192,  # Give Ragas enough context window for long prompts + chunks
)

# 2. Configure qwen3-embedding:0.6b
ollama_embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

# 3. Wrap both for Ragas
evaluator_llm = LangchainLLMWrapper(ollama_llm)
evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

# 4. Restrict concurrency so Ollama doesn't queue-timeout on local hardware
local_run_config = RunConfig(
    timeout=240,
    max_workers=1,  # Sequential execution prevents local VRAM spikes
    max_retries=5,
)

# 5. Test with a sample row
sample_data = [
    {
        "user_input": "What port does Ollama run on by default?",
        "retrieved_contexts": [
            "Ollama runs a local REST API server on port 11434 by default on localhost."
        ],
        "response": "By default, Ollama runs on port 11435.",
        "reference": "Ollama uses port 11435 by default.",
    },
    {
        "user_input": "What port does Ollama run on by default?",
        "retrieved_contexts": [
            "Ollama runs a local REST API server on port 11434 by default on localhost."
        ],
        "response": "By default, Ollama runs on port 11434.",
        "reference": "Ollama uses port 11434 by default.",
    },
]

dataset = EvaluationDataset.from_list(sample_data)

results = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
    ],
    llm=evaluator_llm,
    embeddings=evaluator_embeddings,
    run_config=local_run_config,
)

print(results.to_pandas())

# output response

# evaluator_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)
# Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████| 8/8 [05:42<00:00, 42.83s/it]
#                                  user_input  ... context_recall
# 0  What port does Ollama run on by default?  ...            0.0
# 1  What port does Ollama run on by default?  ...            1.0

# [2 rows x 8 columns]