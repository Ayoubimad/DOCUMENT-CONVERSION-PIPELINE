import sys
from pathlib import Path
import json
import asyncio
from typing import List, Dict
import os
import multiprocessing as mp

sys.path.append(str(Path(__file__).parent.parent))
from chunking import AgenticChunking
from document import Document
from llm import OpenAIModel

NUM_CORES = os.cpu_count()

openai_model = OpenAIModel(
    model="casperhansen/mistral-small-24b-instruct-2501-awq",
    temperature=0.0,
    max_tokens=8192,
    base_url="http://172.18.21.138:8000/v1",
    api_key="random_api_key",
)

agentic_chunker = AgenticChunking(model=openai_model, max_chunk_size=10000)


def generate_questions_for_chunk(chunk: Document, num_questions: int = 3) -> List[str]:
    """Generate questions that can be answered using the chunk content."""
    prompt = f"""Given the following text, generate {num_questions} questions following these STRICT requirements:

    REQUIREMENTS:
    1. Each question MUST be answerable using ONLY the provided text
    2. Each question MUST ask about information that IS EXPLICITLY stated in the text
    3. **Questions MUST be in the same language as the provided text**
    4. Questions MUST be completely different from each other
    5. Each question should focus on a different aspect or piece of information
    6. Questions MUST be clear and specific
    7. Questions MUST be in proper question format (not statements)
    8. Avoid yes/no questions - focus on what, how, why, when, where, who questions
    9. Questions should test understanding of key concepts, not trivial details

    FORMAT:
    - Return ONLY the questions
    - One question per line
    - No numbering or bullet points
    - No explanations or additional text

    Text to generate questions from:
    {chunk.content}
    """

    response = openai_model.generate(prompt)
    questions = [q.strip() for q in response.strip().split("\n") if q.strip()]
    return questions[:num_questions]


async def process_chunk(
    chunk: Document, chunk_index: int, total_chunks: int, doc_name: str
) -> Dict:
    """Process a single chunk to generate questions and references."""
    print(f"Processing chunk {chunk_index}/{total_chunks} from {doc_name}")

    questions = generate_questions_for_chunk(chunk)
    return {"questions": questions, "references": [chunk.content] * len(questions)}


async def process_document(doc_path: str) -> Dict:
    """Process a document to generate chunks and questions."""
    print(f"Processing document: {doc_path}")
    with open(doc_path, "r") as f:
        document = Document(content=f.read(), name=Path(doc_path).name)

    chunks = await agentic_chunker.chunk_async(document)
    print(f"Generated {len(chunks)} chunks for {document.name}")

    chunk_tasks = []
    for i, chunk in enumerate(chunks, 1):
        chunk_tasks.append(process_chunk(chunk, i, len(chunks), document.name))

    chunk_results = await asyncio.gather(*chunk_tasks)

    document_dataset = {"questions": [], "references": []}
    for result in chunk_results:
        document_dataset["questions"].extend(result["questions"])
        document_dataset["references"].extend(result["references"])

    return document_dataset


def process_document_wrapper(doc_path):
    """Wrapper to run the async process_document function in a separate process."""
    return asyncio.run(process_document(doc_path))


async def create_evaluation_dataset(document_paths: List[str], output_path: str):
    """Create an evaluation dataset from the given documents."""
    dataset = {"questions": [], "references": []}

    with mp.Pool(processes=NUM_CORES) as pool:
        doc_results = pool.map(process_document_wrapper, document_paths)

    for doc_dataset in doc_results:
        dataset["questions"].extend(doc_dataset["questions"])
        dataset["references"].extend(doc_dataset["references"])

    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


def main():
    doc_paths = []

    for file in Path("/home/e4user/document-conversion-pipeline/data/output").glob(
        "*.md"
    ):
        doc_paths.append(str(file))

    output_path = "rag_evaluation_dataset.json"
    dataset = asyncio.run(create_evaluation_dataset(doc_paths, output_path))

    print(
        f"\nGenerated dataset with {len(dataset['questions'])} question-reference pairs"
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
