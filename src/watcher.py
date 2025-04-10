"""
Document watcher module that monitors input directories for new files, detects when
files are fully written, processes them using a document converter, and saves the
results to an output directory.
"""

import os
import queue
import threading
import time
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List, Set
from collections import defaultdict

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter
from document import Document
from config import settings

# Performance tuning constants
STABILITY_CHECKS = 3  # Reduced from 10 to 3 for faster processing
STABILITY_INTERVAL = 0.1  # Reduced from 0.2 to 0.1 seconds
MAX_WAIT_TIME = 3600  # Reduced from 86400 to 3600 seconds (1 hour)
MIN_GROWTH_RATE = 1024  # Minimum growth rate in bytes/second
BATCH_SIZE = 32  # Number of files to process in a batch
BATCH_TIMEOUT = 2.0  # Maximum time to wait for batch completion
MAX_WORKERS = os.cpu_count() or 1  # Maximum number of worker threads based on CPU cores

print(f"MAX_WORKERS: {MAX_WORKERS}")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document file system events with batch processing support."""

    def __init__(
        self,
        converter: DocumentConverter,
        output_dir: str,
    ):
        """Initialize the document event handler."""
        self.converter = converter
        self.output_dir = output_dir
        self.queue: queue.Queue = queue.Queue()
        self.ready_queue: queue.Queue = queue.Queue()
        self.is_running = True
        self.processor_thread: Optional[threading.Thread] = None
        self.file_sizes: Dict[str, Tuple[int, float, int, float]] = {}
        self.batch_lock = threading.Lock()
        self.batch_event = threading.Event()
        self.current_batch: Set[str] = set()
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        self.processing_files: Dict[str, asyncio.Task] = {}

    async def process_batch_async(self, batch: List[str]) -> None:
        """Process a batch of files asynchronously.

        Args:
            batch: List of file paths to process
        """
        try:
            tasks = []
            for file_path in batch:
                if not Path(file_path).exists():
                    continue
                task = asyncio.create_task(self._process_file_async(file_path))
                self.processing_files[file_path] = task
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)

        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
        finally:
            for file_path in batch:
                self.processing_files.pop(file_path, None)

    def process_queue(self) -> None:
        """Continuously process files from the queue with batch support."""
        current_batch: List[str] = []
        last_batch_time = time.time()

        while self.is_running:
            try:
                try:
                    while len(current_batch) < BATCH_SIZE:
                        file_path = self.ready_queue.get_nowait()
                        current_batch.append(file_path)
                        self.ready_queue.task_done()
                except queue.Empty:
                    pass

                current_time = time.time()
                if len(current_batch) >= BATCH_SIZE or (
                    current_batch and current_time - last_batch_time >= BATCH_TIMEOUT
                ):
                    if current_batch:
                        logger.info(f"Processing batch of {len(current_batch)} files")
                        self._process_batch(current_batch)
                        current_batch = []
                        last_batch_time = current_time

                try:
                    file_path = self.queue.get(timeout=1)
                    if self._is_file_stable(file_path):
                        self.ready_queue.put(file_path)
                    else:
                        self.queue.put(file_path)
                    self.queue.task_done()
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Error in queue processor: {e}", exc_info=True)
                current_batch = []

    def _process_batch(self, batch: List[str]) -> None:
        """Process a batch of files using the thread pool."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.process_batch_async(batch))
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
        finally:
            for file_path in batch:
                if file_path in self.file_sizes:
                    del self.file_sizes[file_path]

    def _is_file_stable(self, file_path: str) -> bool:
        """Check if a file has finished being written using optimized stability detection."""
        try:
            if not os.path.exists(file_path):
                return False

            current_size = os.path.getsize(file_path)
            current_time = time.time()

            if current_size == 0:
                return False

            if file_path not in self.file_sizes:
                self.file_sizes[file_path] = (
                    current_size,
                    current_time,
                    0,
                    current_time,
                )
                return False

            prev_size, first_seen_time, stable_count, last_change_time = (
                self.file_sizes[file_path]
            )

            if current_size < 1024 * 1024 and current_size == prev_size:  # 1MB
                return True

            if current_time - first_seen_time > MAX_WAIT_TIME:
                return True

            if current_size == prev_size:
                stable_count += 1
                self.file_sizes[file_path] = (
                    current_size,
                    first_seen_time,
                    stable_count,
                    last_change_time,
                )
                return stable_count >= STABILITY_CHECKS

            self.file_sizes[file_path] = (
                current_size,
                first_seen_time,
                0,
                current_time,
            )
            return False

        except Exception as e:
            logger.error(f"Error checking file stability: {e}", exc_info=True)
            return False

    async def _process_file_async(self, file_path: str) -> None:
        """Process a single file using the converter asynchronously."""
        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Converting {file_path} ({file_size/1024/1024:.2f} MB)")

            document = await self.converter.convert_async(file_path)
            await self._save_document(document, file_path)

        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}", exc_info=True)

    async def _save_document(self, document: Document, file_path: str) -> None:
        """Save the converted document to the output directory."""
        try:
            input_path = Path(file_path)
            output_filename = f"{input_path.stem}.md"
            output_path = Path(self.output_dir) / output_filename

            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(document.content, encoding="utf-8")

            logger.info(f"Saved {file_path} to {output_path}")

        except Exception as e:
            logger.error(f"Error saving document from {file_path}: {e}", exc_info=True)

    def start_processing(self) -> None:
        """Start the queue processor in a separate thread."""
        self.processor_thread = threading.Thread(target=self.process_queue)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        logger.info("File processing thread started")

    def stop(self) -> None:
        """Stop the queue processor and clean up resources."""
        logger.info("Stopping file processing")
        self.is_running = False
        self.file_sizes.clear()

        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)

        self.thread_pool.shutdown(wait=False)

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = event.src_path
        if not os.path.exists(file_path):
            return

        logger.debug(f"Queueing new file: {file_path}")
        self.queue.put(file_path)


class DocumentWatcher:
    """Watches a directory for document files and processes them as they arrive."""

    def __init__(
        self,
        converter: DocumentConverter,
        input_dir: str,
        output_dir: str,
    ):
        """Initialize the document watcher.

        Args:
            converter: The converter to use for document processing
            input_dir: The input directory to watch for new documents
            output_dir: The output directory to save converted documents

        Raises:
            ValueError: If input_dir does not exist or is not a directory
        """

        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.converter = converter
        self.input_dir = str(input_path)
        self.output_dir = str(output_path)
        self.observer = Observer()
        self.event_handler = DocumentEventHandler(self.converter, self.output_dir)
        self.is_running = False

    def process_existing_files(self, process_all: bool = False) -> None:
        """Process files that already exist in the input directory.

        This is useful when starting the watcher to process files that are already
        in the directory before the watcher was started.

        Args:
            process_all: If True, process all files. If False, only process files
                         that don't have a corresponding output file.
        """
        logger.info(f"Checking for existing files in {self.input_dir}")

        input_path = Path(self.input_dir)
        output_path = Path(self.output_dir)

        input_files = [f for f in input_path.glob("*") if f.is_file()]

        if not input_files:
            logger.info("No existing files found in input directory")
            return

        logger.info(f"Found {len(input_files)} existing files in input directory")

        files_to_process = []
        for input_file in input_files:
            output_file = output_path / f"{input_file.stem}.md"

            if process_all or not output_file.exists():
                files_to_process.append(str(input_file))

        if files_to_process:
            logger.info(f"Queueing {len(files_to_process)} files for processing")
            for file_path in files_to_process:
                self.event_handler.queue.put(file_path)
        else:
            logger.info("No files need processing")

    def start(self, process_existing: bool = False) -> None:
        """Start watching the input directory and processing queue.

        Args:
            process_existing: Whether to process existing files in the input directory

        Raises:
            RuntimeError: If the watcher is already running
        """
        if self.is_running:
            raise RuntimeError("DocumentWatcher is already running")

        try:
            self.event_handler.start_processing()

            if process_existing:
                self.process_existing_files()

            self.observer.schedule(
                self.event_handler, path=self.input_dir, recursive=True
            )
            self.observer.start()

            self.is_running = True
            logger.info(f"Started watching directory: {self.input_dir}")
            logger.info(f"Converted documents will be saved to: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to start document watcher: {e}", exc_info=True)
            self.stop()
            raise

    def stop(self) -> None:
        """Stop watching the input directory and queue processing."""
        if not self.is_running:
            logger.debug("DocumentWatcher is not running, nothing to stop")
            return

        logger.info(f"Stopping document watcher for {self.input_dir}")

        try:
            self.event_handler.stop()
        except Exception as e:
            logger.error(f"Error stopping event handler: {e}", exc_info=True)

        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            if self.observer.is_alive():
                logger.warning("Observer thread did not terminate cleanly")
        except Exception as e:
            logger.error(f"Error stopping observer: {e}", exc_info=True)

        self.is_running = False
        logger.info(f"Stopped watching {self.input_dir}")
