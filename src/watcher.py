"""
Document watcher module that monitors input directories for new files, detects when
files are fully written, processes them using a document converter, and saves the
results to an output directory.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Dict, Set, Optional, Any
import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter
from document import Document
from config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document file system events with async processing support."""

    def __init__(
        self,
        converter: DocumentConverter,
        output_dir: str,
    ):
        """Initialize the document event handler."""
        self.converter = converter
        self.output_dir = output_dir
        self.is_running = True
        self.pending_files: Set[str] = set()
        self.file_sizes: Dict[str, Dict[str, Any]] = {}
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    async def process_file(self, file_path: str) -> None:
        """Process a file asynchronously with retry mechanism.

        Args:
            file_path: Path to the file to process
        """
        if not Path(file_path).exists():
            logger.warning(f"File doesn't exist anymore: {file_path}")
            self.pending_files.discard(file_path)
            return

        logger.info(f"Processing file: {file_path}")

        try:
            file_size = os.path.getsize(file_path)
            logger.info(f"Converting {file_path} ({file_size/1024/1024:.2f} MB)")

            # Use the convert_async method which has its own retry mechanism
            document = await self.converter.convert_async(
                file_path,
                max_retries=settings.MAX_RETRIES,
                retry_delay=settings.RETRY_DELAY,
            )
            await self._save_document(document, file_path)

            # Successful conversion
            self.pending_files.discard(file_path)
            if file_path in self.file_sizes:
                del self.file_sizes[file_path]

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            # The file will be removed from pending files if all retries fail in converter
            self.pending_files.discard(file_path)
            if file_path in self.file_sizes:
                del self.file_sizes[file_path]

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
            raise

    async def check_file_stability(self, file_path: str) -> bool:
        """Check if a file has finished being written using simplified stability detection.

        Args:
            file_path: Path to check for stability

        Returns:
            bool: True if the file is stable, False otherwise
        """
        if not os.path.exists(file_path):
            return False

        current_size = os.path.getsize(file_path)
        current_time = time.time()

        if current_size == 0:
            return False

        if file_path not in self.file_sizes:
            self.file_sizes[file_path] = {
                "size": current_size,
                "first_seen": current_time,
                "stable_count": 0,
                "last_check": current_time,
            }
            return False

        file_data = self.file_sizes[file_path]

        if current_size == file_data["size"]:
            file_data["stable_count"] += 1
            file_data["last_check"] = current_time

            if current_size < 1024 * 1024:
                return True

            return file_data["stable_count"] >= settings.STABILITY_CHECKS

        file_data["size"] = current_size
        file_data["stable_count"] = 0
        file_data["last_check"] = current_time

        if current_time - file_data["first_seen"] > settings.MAX_WAIT_TIME:
            logger.warning(f"File {file_path} considered stable after max wait time")
            return True

        return False

    async def monitor_pending_files(self) -> None:
        """Continuously monitor pending files and process them when stable."""
        while self.is_running:
            try:
                current_pending = self.pending_files.copy()

                for file_path in current_pending:
                    if (
                        file_path in self.processing_tasks
                        and not self.processing_tasks[file_path].done()
                    ):
                        continue

                    is_stable = await self.check_file_stability(file_path)
                    if is_stable:
                        task = asyncio.create_task(self.process_file(file_path))
                        self.processing_tasks[file_path] = task

                        self._clean_completed_tasks()

                await asyncio.sleep(settings.STABILITY_INTERVAL)

            except Exception as e:
                logger.error(f"Error in pending file monitor: {e}", exc_info=True)
                await asyncio.sleep(1)

    def _clean_completed_tasks(self) -> None:
        """Remove completed tasks from the tracking dictionary."""
        completed_tasks = [
            file_path
            for file_path, task in self.processing_tasks.items()
            if task.done()
        ]

        for file_path in completed_tasks:
            task = self.processing_tasks.pop(file_path)
            if task.exception():
                logger.error(
                    f"Task for {file_path} raised an exception: {task.exception()}"
                )

    def start_async(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the async file monitor."""
        self.loop = loop
        self.is_running = True
        self.loop.create_task(self.monitor_pending_files())
        logger.info("Async file monitor started")

    def stop(self) -> None:
        """Stop the file monitor."""
        logger.info("Stopping file processing")
        self.is_running = False
        self.file_sizes.clear()

    def on_created(self, event: Any) -> None:
        """Handle file creation events."""
        if event.is_directory:
            return

        file_path = event.src_path
        if not os.path.exists(file_path):
            return

        logger.debug(f"Queueing new file: {file_path}")
        self.pending_files.add(file_path)

    def on_modified(self, event: Any) -> None:
        """Handle file modification events."""
        if event.is_directory:
            return

        file_path = event.src_path
        if not os.path.exists(file_path):
            return

        if file_path not in self.pending_files:
            logger.debug(f"Queueing modified file: {file_path}")
            self.pending_files.add(file_path)


class DocumentWatcher:
    """Watches a directory for document files and processes them."""

    def __init__(
        self,
        converter: DocumentConverter,
        input_dir: str,
        output_dir: str,
    ):
        """Initialize the document watcher.

        Args:
            converter: The document converter to use
            input_dir: The directory to watch for documents
            output_dir: The directory to save converted documents
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.event_handler = DocumentEventHandler(converter, output_dir)
        self.observer = Observer()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

    async def process_existing_files_async(self) -> None:
        """Process existing files in the input directory asynchronously."""
        input_path = Path(self.input_dir)
        if not input_path.exists() or not input_path.is_dir():
            logger.warning(f"Input directory does not exist: {self.input_dir}")
            return

        files = [str(f) for f in input_path.glob("*") if f.is_file()]
        if not files:
            logger.info("No existing files found in input directory")
            return

        logger.info(f"Processing {len(files)} existing files")

        for file_path in files:
            self.event_handler.pending_files.add(file_path)

    def process_existing_files(self, process_all: bool = False) -> None:
        """Process existing files in the input directory."""
        if self.loop:
            asyncio.run_coroutine_threadsafe(
                self.process_existing_files_async(), self.loop
            )

    def start(self, process_existing: bool = False) -> None:
        """Start watching the input directory.

        Args:
            process_existing: Whether to process existing files in the input directory
        """
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.event_handler.start_async(self.loop)

        self.observer.schedule(self.event_handler, self.input_dir, recursive=False)
        self.observer.start()
        logger.info(f"Started watching directory: {self.input_dir}")

        if process_existing:
            self.process_existing_files(process_all=True)

    def stop(self) -> None:
        """Stop watching the input directory."""
        if self.observer.is_alive():
            self.observer.stop()
            self.observer.join()

        self.event_handler.stop()
        logger.info("Stopped watching directory")
