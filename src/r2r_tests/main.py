import json
import asyncio
from typing import List
from ragas.integrations.r2r import transform_to_ragas_dataset
import pickle
import logging
import colorlog
from .config import RAGConfig, RAGGenerationConfig, SearchSettings
from .r2r_tester import RAGTester
from .metrics import MetricsEvaluator, MetricsConfig


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

colors = {
    "DEBUG": "cyan",
    "INFO": "green",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red,bg_white",
}

formatter = colorlog.ColoredFormatter(
    "%(asctime)s - %(log_color)s%(levelname)s%(reset)s - %(message)s",
    log_colors=colors,
    reset=True,
    style="%",
)

console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


async def run_rag_evaluation(
    dataset_path: str,
    config: RAGConfig,
    metrics_config: MetricsConfig,
    output_path: str = "ragas_eval_dataset",
) -> None:
    """Main async function to run the complete RAG evaluation"""
    logger.info(f"Starting evaluation with config: {config}")

    async with RAGTester() as tester:
        logger.info(f"Loading dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            dataset = json.load(f)

        questions = dataset["questions"]
        references = dataset["references"]
        unique_chunks = list(set(references))

        logger.info(
            f"Dataset loaded: {len(questions)} questions, {len(unique_chunks)} unique chunks"
        )

        logger.info("Cleaning up existing documents...")
        await tester.delete_all_documents()

        logger.info("Starting chunk ingestion...")
        await tester.ingest_chunks(unique_chunks)

        logger.info("Processing RAG queries...")
        r2r_responses = await tester.process_rag_queries(questions, config)

        logger.info("Transforming results to Ragas format...")
        ragas_eval_dataset = transform_to_ragas_dataset(
            user_inputs=questions, r2r_responses=r2r_responses, references=references
        )

        evaluator = MetricsEvaluator(metrics_config)
        results = evaluator.evaluate_dataset(ragas_eval_dataset)

        logger.info(f"Evaluation results: {results}")


async def test_rag_configuration(
    dataset_path: str, config: RAGConfig, metrics_config: MetricsConfig
):
    """Test a single RAG configuration asynchronously"""
    await run_rag_evaluation(
        dataset_path=dataset_path,
        config=config,
        metrics_config=metrics_config,
        output_path=f"ragas_eval_dataset_{config.name}",
    )


if __name__ == "__main__":

    generation_config = RAGGenerationConfig(
        model="deepseek/ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        api_base="http://172.18.21.137:8000/v1",
        temperature=0.0,
        max_tokens=8192,
    )

    search_settings = SearchSettings(
        limit=3,
        graph_settings={"enabled": False},
    )

    rag_config = RAGConfig(
        generation_config=generation_config,
        search_settings=search_settings,
        search_mode="basic",
        include_web_search=False,
    )

    metrics_config = MetricsConfig(
        llm_model="ISTA-DASLab/gemma-3-27b-it-GPTQ-4b-128g",
        llm_api_base="http://172.18.21.137:8000/v1",
        embeddings_model="BAAI/bge-m3",
        embeddings_api_base="http://172.18.21.126:8000/v1",
    )

    asyncio.run(
        test_rag_configuration(
            dataset_path="/home/e4user/document-conversion-pipeline/src/r2r_tests/rag_evaluation_dataset.json",
            config=rag_config,
            metrics_config=metrics_config,
        )
    )
