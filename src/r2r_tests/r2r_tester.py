import os
import asyncio
from typing import List
from r2r import R2RClient
import logging
import colorlog
import aiohttp
from concurrent.futures import ThreadPoolExecutor
from .config import RAGConfig


"""
Using ThreadPoolExecutor makes blocking I/O operations faster in asyncio by running them in separate threads, 
preventing them from blocking the event loop.
The R2R client uses synchronous HTTP calls, which would normally block the entire asyncio event loop while waiting for responses. 
By running these operations in a ThreadPoolExecutor with loop.run_in_executor(), 
they execute in background threads while the event loop continues handling other tasks.
This allows us to:
- Make multiple API calls concurrently
- Process other tasks while waiting for network responses
- Handle more queries simultaneously without getting blocked
"""


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


class RAGTester:
    def __init__(self, r2r_url: str = "http://localhost:7272"):
        self.r2r_url = r2r_url
        self.client = R2RClient(r2r_url, timeout=60000)
        self.session = None
        self.executor = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300))
        self.executor = ThreadPoolExecutor(max_workers=os.cpu_count())
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
        if self.executor:
            self.executor.shutdown(wait=True)

    async def delete_all_documents(self) -> None:
        """Delete all documents from R2R asynchronously"""
        try:
            documents = self.client.documents.list()
            logger.info(f"Found {len(documents.results)} documents to delete")

            tasks = [self.delete_document(doc.id) for doc in documents.results]
            await asyncio.gather(*tasks)
            logger.info("All documents deleted")
        except Exception as e:
            logger.error(f"Error during delete_all_documents: {e}")

    async def delete_document(self, doc_id: str) -> None:
        """Delete a single document asynchronously with retry logic"""
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            self.executor, self.client.documents.delete, str(doc_id)
        )
        logger.debug(f"Deleted document: {doc_id}")
        return

    async def ingest_chunks(self, chunks: List[str], batch_size: int = 50) -> None:
        """Ingest chunks into R2R in batches asynchronously"""
        total_chunks = len(chunks)
        logger.info(f"Starting ingestion of {total_chunks} chunks")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    self.executor, lambda: self.client.documents.create(chunks=batch)
                )
                logger.info(
                    f"Ingested batch {i//batch_size + 1}/{(total_chunks+batch_size-1)//batch_size}"
                )
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Error ingesting chunk batch {i//batch_size + 1}: {e}")

    async def process_rag_queries(
        self, questions: List[str], config: RAGConfig
    ) -> List[str]:
        """Process RAG queries with controlled concurrency"""
        total_questions = len(questions)
        logger.info(f"Processing {total_questions} questions")

        semaphore = asyncio.Semaphore(100)

        async def process_with_semaphore(i, question):
            async with semaphore:
                return await self.process_query(i, question, config, total_questions)

        tasks = [
            process_with_semaphore(i, question) for i, question in enumerate(questions)
        ]

        responses = await asyncio.gather(*tasks)

        if logger.getEffectiveLevel() == logging.DEBUG:
            for i, (question, response) in enumerate(zip(questions, responses)):
                if (i + 1) % 10 == 0:
                    logger.debug(f"\n=== Sample Response for Question {i+1} ===")
                logger.debug(f"Question: {question}")
                logger.debug(f"Response: {response}")
                logger.debug("=====================================\n")

        return responses

    async def process_query(
        self, index: int, question: str, config: RAGConfig, total: int
    ) -> str:
        """Process a single RAG query asynchronously with retries"""
        logger.debug(f"Processing question {index+1}/{total}: {question[:50]}...")
        response = await self.run_in_executor(
            self.client.retrieval.rag, query=question, **config.to_rag_params()
        )
        logger.info(f"Processed question {index+1}/{total}")
        return response

    async def run_in_executor(self, func, *args, **kwargs):
        """Helper function to run synchronous code in the executor"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, lambda: func(*args, **kwargs))
