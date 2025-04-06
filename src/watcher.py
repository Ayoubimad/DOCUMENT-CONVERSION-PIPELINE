"""
Document watcher module that monitors input directories for new files, detects when
files are fully written, processes them using a document converter, and saves the
results to an output directory. Includes file stability detection, statistics tracking,
and async processing capabilities.
"""

import os
import queue
import threading
import time
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter
from document import Document
from config import settings

# Constants for file stability detection
STABILITY_CHECKS = 10  # Number of consecutive stable checks required
STABILITY_INTERVAL = 0.2  # Time between checks in seconds
MAX_WAIT_TIME = 86400  # Maximum time to wait for file stability (seconds)
MIN_GROWTH_RATE = (
    1024  # Minimum growth rate in bytes/second to consider a file still being written
)
STATS_SAVE_INTERVAL = 300  # Save stats every 5 minutes (seconds)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)

logger = logging.getLogger(__name__)


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document file system events.

    Detects new or modified files and processes them using the provided document converter.
    """

    def __init__(
        self,
        converter: DocumentConverter,
        output_dir: str,
    ):
        """Initialize the document event handler.

        Args:
            converter: The converter to use for document processing
            output_dir: The output directory to save converted documents

        """
        self.converter = converter
        self.output_dir = output_dir
        self.queue: queue.Queue = queue.Queue()  # Queue for files to check stability
        self.ready_queue: queue.Queue = (
            queue.Queue()
        )  # Queue for stable files ready to process
        self.is_running = True
        self.processor_thread: Optional[threading.Thread] = None
        self.file_sizes: Dict[str, Tuple[int, float, int, float]] = {}

    def process_queue(self) -> None:
        """Continuously process files from the queue until stopped."""
        while self.is_running:
            try:
                # First try to process any ready files
                try:
                    file_path = self.ready_queue.get_nowait()
                    logger.info(f"Processing ready file: {file_path}")
                    self._run_process_file(file_path)
                    if file_path in self.file_sizes:
                        del self.file_sizes[file_path]
                    self.ready_queue.task_done()
                    continue
                except queue.Empty:
                    pass

                # Check stability of files
                try:
                    file_path = self.queue.get(timeout=1)
                    logger.debug(f"Checking stability of file: {file_path}")
                    if self._is_file_stable(file_path):
                        logger.info(
                            f"File is stable, moving to ready queue: {file_path}"
                        )
                        self.ready_queue.put(file_path)
                    else:
                        logger.debug(f"File not yet stable, re-queueing: {file_path}")
                        self.queue.put(file_path)
                    self.queue.task_done()
                except queue.Empty:
                    continue

            except Exception as e:
                logger.error(f"Error processing file in queue: {e}", exc_info=True)

    def _is_file_stable(self, file_path: str) -> bool:
        """Check if a file has finished being written by monitoring size stability.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file size has stabilized, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.info(f"File {file_path} does not exist")
                return False

            current_size = os.path.getsize(file_path)
            current_time = time.time()

            if current_size == 0:
                logger.debug(f"File {file_path} has zero size")
                return False

            if file_path not in self.file_sizes:
                logger.debug(
                    f"First time seeing file {file_path}, size: {current_size}"
                )
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

            if current_time - first_seen_time > MAX_WAIT_TIME:
                logger.info(
                    f"File {file_path} exceeded max wait time, marking as stable"
                )
                return True

            if current_size == prev_size:
                stable_count += 1
                logger.debug(
                    f"File {file_path} stable check {stable_count}/{STABILITY_CHECKS}"
                )

                self.file_sizes[file_path] = (
                    current_size,
                    first_seen_time,
                    stable_count,
                    last_change_time,
                )

                if stable_count >= STABILITY_CHECKS:
                    logger.info(
                        f"File {file_path} is stable after {stable_count} checks"
                    )
                    return True

                time.sleep(STABILITY_INTERVAL)
                return False
            else:
                logger.debug(
                    f"File {file_path} size changed from {prev_size} to {current_size}"
                )
                self.file_sizes[file_path] = (
                    current_size,
                    first_seen_time,
                    0,  # Reset stable count
                    current_time,
                )
                return False

        except (FileNotFoundError, PermissionError) as e:
            logger.info(
                f"File access error during stability check for {file_path}: {e}"
            )
            return False
        except Exception as e:
            logger.error(
                f"Error checking file stability for {file_path}: {e}", exc_info=True
            )
            return False

    async def _process_file_async(self, file_path: str) -> None:
        """Process a single file using the converter asynchronously.

        Args:
            file_path: Path to the file to be processed.
        """

        try:
            file_size = os.path.getsize(file_path)
            logger.info(
                f"Starting async conversion of {file_path} ({file_size/1024/1024:.2f} MB)"
            )

            try:
                document = await self.converter.convert_async(file_path)
                """
                document = await asyncio.wait_for(
                    self.converter.convert_async(file_path),
                    timeout=self.converter.client.timeout,
                )
                """
                await self._save_document(document, file_path)

            except asyncio.TimeoutError:
                logger.error(
                    f"Conversion timeout for {file_path} after {self.converter.client.timeout} seconds"
                )
                raise

        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}", exc_info=True)

    async def _save_document(self, document: Document, file_path: str) -> None:
        """Save the converted document to the output directory.

        Args:
            document: The converted document to save
            file_path: Original input file path for reference
        """
        try:

            input_path = Path(file_path)
            output_filename = f"{input_path.stem}.md"
            output_path = Path(self.output_dir) / output_filename

            output_path.parent.mkdir(parents=True, exist_ok=True)

            output_path.write_text(document.content, encoding="utf-8")

            logger.info(
                f"Successfully converted and saved {file_path} to {output_path}"
            )

        except IOError as e:
            logger.error(
                f"Error writing output file for {file_path}: {e}", exc_info=True
            )
        except Exception as e:
            logger.error(
                f"Unexpected error saving document from {file_path}: {e}", exc_info=True
            )

    def _run_process_file(self, file_path: str) -> None:
        """Run the async process_file method in the appropriate event loop.

        Args:
            file_path: Path to the file to be processed.
        """
        if not Path(file_path).exists():
            logger.warning(f"File {file_path} no longer exists, skipping processing")
            return

        try:
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            if loop.is_running():
                logger.debug(f"Using run_coroutine_threadsafe for {file_path}")
                future = asyncio.run_coroutine_threadsafe(
                    self._process_file_async(file_path), loop
                )
                try:
                    future.result(timeout=settings.DOCLING_TIMEOUT + 60)
                except asyncio.TimeoutError:
                    logger.error(
                        f"Conversion timeout for {file_path} after {settings.DOCLING_TIMEOUT + 60} seconds"
                    )
                except Exception as e:
                    logger.error(
                        f"Error in async processing of {file_path}: {e}", exc_info=True
                    )
            else:
                logger.debug(f"Using run_until_complete for {file_path}")
                try:
                    loop.run_until_complete(self._process_file_async(file_path))
                except asyncio.TimeoutError:
                    logger.error(
                        f"Conversion timeout for {file_path} after {settings.DOCLING_TIMEOUT} seconds"
                    )
                except Exception as e:
                    logger.error(f"Error in processing {file_path}: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Error setting up async environment for {file_path}: {e}",
                exc_info=True,
            )

    def start_processing(self) -> None:
        """Start the queue processor in a separate thread."""
        self.processor_thread = threading.Thread(target=self.process_queue)
        self.processor_thread.daemon = True
        self.processor_thread.start()
        logger.info("File processing thread started")

    def stop(self) -> None:
        """Stop the queue processor."""
        logger.info("Stopping file processing thread")
        self.is_running = False
        self.file_sizes.clear()

        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
            if self.processor_thread.is_alive():
                logger.warning("File processing thread did not terminate cleanly")

    def on_created(self, event: Any) -> None:
        """Handle file creation events.

        Args:
            event: The file system event
        """
        if event.is_directory:
            logger.debug(f"Ignoring directory creation: {event.src_path}")
            return

        logger.info(
            f"New file detected, waiting for filesystem to stabilize: {event.src_path}"
        )
        time.sleep(0.5)  # Brief delay to let the file system stabilize

        file_path = event.src_path
        if not os.path.exists(file_path):
            logger.warning(f"File no longer exists after delay: {file_path}")
            return

        logger.info(f"Queueing new file for processing: {file_path}")
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
