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
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List, Set

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter
from document import Document


# Constants for file stability detection
STABILITY_CHECKS = 3  # Number of consecutive stable checks required
STABILITY_INTERVAL = 0.5  # Time between checks in seconds
MAX_WAIT_TIME = 30  # Maximum time to wait for file stability (seconds)
MIN_GROWTH_RATE = (
    1024  # Minimum growth rate in bytes/second to consider a file still being written
)
CONVERSION_TIMEOUT = 60  # Timeout for document conversion operations (seconds)
STATS_SAVE_INTERVAL = 300  # Save stats every 5 minutes (seconds)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class FileStats:
    """Class for tracking file conversion statistics."""

    def __init__(self, stats_file: Optional[str] = None):
        """Initialize file statistics tracker.

        Args:
            stats_file: Optional path to save statistics
        """
        self.stats_file = stats_file
        self.processed_files: Set[str] = set()
        self.successful_conversions: Set[str] = set()
        self.failed_conversions: Set[str] = set()
        self.file_types: Dict[str, int] = {}
        self.processing_times: Dict[str, float] = {}
        self.last_save_time = time.time()

    def record_processing_start(self, file_path: str) -> None:
        """Record the start of file processing.

        Args:
            file_path: Path of the file being processed
        """
        self.processed_files.add(file_path)

        # Track file type
        extension = Path(file_path).suffix.lower()
        self.file_types[extension] = self.file_types.get(extension, 0) + 1

        # Store start time for duration tracking
        self.processing_times[file_path] = time.time()

    def record_processing_result(self, file_path: str, success: bool) -> None:
        """Record the result of file processing.

        Args:
            file_path: Path of the processed file
            success: Whether processing was successful
        """
        if success:
            self.successful_conversions.add(file_path)
        else:
            self.failed_conversions.add(file_path)

        # Calculate processing duration
        if file_path in self.processing_times:
            start_time = self.processing_times[file_path]
            duration = time.time() - start_time
            self.processing_times[file_path] = duration

        # Save stats periodically
        if self.stats_file and time.time() - self.last_save_time > STATS_SAVE_INTERVAL:
            self.save_stats()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of processing statistics.

        Returns:
            Dict[str, Any]: Summary statistics
        """
        return {
            "total_processed": len(self.processed_files),
            "successful": len(self.successful_conversions),
            "failed": len(self.failed_conversions),
            "file_types": self.file_types,
            "avg_processing_time": self._calculate_avg_time(),
        }

    def _calculate_avg_time(self) -> Dict[str, float]:
        """Calculate average processing time by file type.

        Returns:
            Dict[str, float]: Average processing time by file extension
        """
        # Filter out entries that are still processing (have start time, not duration)
        durations = {
            k: v
            for k, v in self.processing_times.items()
            if isinstance(v, float) and v > 0
        }

        if not durations:
            return {}

        # Group by file extension
        extension_times: Dict[str, List[float]] = {}
        for file_path, time_value in durations.items():
            ext = Path(file_path).suffix.lower()
            if ext not in extension_times:
                extension_times[ext] = []
            extension_times[ext].append(time_value)

        # Calculate averages
        return {ext: sum(times) / len(times) for ext, times in extension_times.items()}

    def save_stats(self) -> None:
        """Save statistics to file if stats_file is configured."""
        if not self.stats_file:
            return

        try:
            summary = self.get_summary()
            summary["timestamp"] = datetime.now().isoformat()

            with open(self.stats_file, "w") as f:
                json.dump(summary, f, indent=2)

            self.last_save_time = time.time()
            logger.debug(f"Saved processing statistics to {self.stats_file}")
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}", exc_info=True)


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document file system events.

    Detects new or modified files and processes them using the provided document converter.
    """

    def __init__(
        self,
        converter: DocumentConverter,
        output_dir: str,
        stats_file: Optional[str] = None,
    ):
        """Initialize the document event handler.

        Args:
            converter: The converter to use for document processing
            output_dir: The output directory to save converted documents
            stats_file: Optional path to save conversion statistics
        """
        self.converter = converter
        self.output_dir = output_dir
        self.queue: queue.Queue = queue.Queue()
        self.is_running = True
        self.processor_thread: Optional[threading.Thread] = None
        self.file_sizes: Dict[str, Tuple[int, float, int, float]] = {}
        self.stats = FileStats(stats_file)

    def process_queue(self) -> None:
        """Continuously process files from the queue until stopped."""
        while self.is_running:
            try:
                file_path = self.queue.get(timeout=1)

                if self._is_file_stable(file_path):
                    self._run_process_file(file_path)
                    if file_path in self.file_sizes:
                        del self.file_sizes[file_path]
                else:
                    # Re-queue the file to check again later
                    self.queue.put(file_path)

                self.queue.task_done()
            except queue.Empty:
                # No files in queue, continue waiting
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
            # Basic file existence and size check
            if not os.path.exists(file_path):
                logger.warning(f"File {file_path} does not exist")
                return False

            current_size = os.path.getsize(file_path)
            current_time = time.time()

            if current_size == 0:
                logger.info(f"File {file_path} has zero size, waiting...")
                return False

            # First check for this file
            if file_path not in self.file_sizes:
                # Store as (size, timestamp, consecutive_stable_checks, last_size_change_time)
                self.file_sizes[file_path] = (
                    current_size,
                    current_time,
                    0,
                    current_time,
                )
                logger.debug(
                    f"Started tracking file {file_path} with size {current_size} bytes"
                )
                return False

            # Get previous tracking data
            prev_size, first_seen_time, stable_count, last_change_time = (
                self.file_sizes[file_path]
            )

            # Check if max wait time exceeded
            if current_time - first_seen_time > MAX_WAIT_TIME:
                logger.info(
                    f"File {file_path} exceeded maximum wait time of {MAX_WAIT_TIME}s, processing anyway"
                )
                return True

            # Case 1: File size unchanged - potentially stable
            if current_size == prev_size:
                stable_count += 1
                logger.debug(
                    f"File {file_path} size stable at {current_size} bytes (check {stable_count}/{STABILITY_CHECKS})"
                )

                # Update tracking data with new timestamp and incremented stable count
                self.file_sizes[file_path] = (
                    current_size,
                    first_seen_time,  # Keep original first seen time
                    stable_count,
                    last_change_time,
                )

                # File is stable after enough consecutive stable checks
                if stable_count >= STABILITY_CHECKS:
                    logger.info(
                        f"File {file_path} is stable after {stable_count} checks"
                    )
                    return True

                # Not enough consecutive checks yet, wait before next check
                time.sleep(STABILITY_INTERVAL)
                return False

            # Case 2: File size changed - reset stability counter
            else:
                logger.debug(
                    f"File {file_path} size changed from {prev_size} to {current_size} bytes"
                )

                # Special case: If file is growing very slowly, it might be near completion
                if current_size > prev_size and (current_time - last_change_time > 2.0):
                    growth_rate = (current_size - prev_size) / (
                        current_time - last_change_time
                    )
                    if growth_rate < MIN_GROWTH_RATE:
                        logger.info(
                            f"File {file_path} growing very slowly ({growth_rate:.2f} bytes/sec), considering it stable"
                        )
                        return True

                # Update tracking data with reset stable count and new change time
                self.file_sizes[file_path] = (
                    current_size,
                    first_seen_time,  # Keep original first seen time
                    0,  # Reset stability counter
                    current_time,  # Update last change time
                )
                time.sleep(STABILITY_INTERVAL)
                return False

        except (FileNotFoundError, PermissionError) as e:
            logger.warning(
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
        self.stats.record_processing_start(file_path)
        success = False

        try:
            logger.info(f"Starting async conversion of {file_path}")

            # Preprocess the file
            if not await self._preprocess_file(file_path):
                logger.warning(
                    f"Preprocessing failed for {file_path}, skipping conversion"
                )
                self.stats.record_processing_result(file_path, False)
                return

            # Convert the document
            document = await self.converter.convert_async(file_path)

            # Save the converted document
            await self._save_document(document, file_path)

            success = True

        except Exception as e:
            logger.error(f"Failed to convert {file_path}: {e}", exc_info=True)
        finally:
            self.stats.record_processing_result(file_path, success)

    async def _preprocess_file(self, file_path: str) -> bool:
        """Preprocess a file before conversion.

        Performs initial validation, file type detection, and any necessary
        preprocessing steps before conversion.

        Args:
            file_path: Path to the file to preprocess

        Returns:
            bool: True if preprocessing succeeded, False otherwise
        """
        try:
            file_path_obj = Path(file_path)

            # Check if file still exists
            if not file_path_obj.exists():
                logger.warning(f"File no longer exists: {file_path}")
                return False

            # Check file size
            file_size = file_path_obj.stat().st_size
            if file_size == 0:
                logger.warning(f"File is empty: {file_path}")
                return False

            # Check file extension for supported types
            # This could be expanded based on the converter's capabilities
            extension = file_path_obj.suffix.lower()
            supported_extensions = [
                ".pdf",
                ".docx",
                ".doc",
                ".txt",
                ".rtf",
                ".md",
                ".html",
            ]

            if extension not in supported_extensions:
                logger.warning(
                    f"File type {extension} may not be supported: {file_path}"
                )
                # Not returning False here, still attempt conversion

            logger.debug(
                f"Preprocessing successful for {file_path} ({file_size} bytes, type: {extension})"
            )
            return True

        except Exception as e:
            logger.error(f"Error preprocessing file {file_path}: {e}", exc_info=True)
            return False

    async def _save_document(self, document: Document, file_path: str) -> None:
        """Save the converted document to the output directory.

        Args:
            document: The converted document to save
            file_path: Original input file path for reference
        """
        try:
            # Create output path with consistent Path objects
            input_path = Path(file_path)
            output_filename = f"{input_path.stem}.md"
            output_path = Path(self.output_dir) / output_filename

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the document content to file
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
            # Create a new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # No event loop in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            # Run the async function using the appropriate method
            if loop.is_running():
                # If we're in an async context, create a future
                logger.debug(f"Using run_coroutine_threadsafe for {file_path}")
                future = asyncio.run_coroutine_threadsafe(
                    self._process_file_async(file_path), loop
                )
                try:
                    # Wait for the result with a timeout
                    future.result(timeout=CONVERSION_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.error(
                        f"Conversion timeout for {file_path} after {CONVERSION_TIMEOUT} seconds"
                    )
                except Exception as e:
                    logger.error(
                        f"Error in async processing of {file_path}: {e}", exc_info=True
                    )
            else:
                # If we're not in an async context, we can use run_until_complete
                logger.debug(f"Using run_until_complete for {file_path}")
                try:
                    loop.run_until_complete(self._process_file_async(file_path))
                except asyncio.TimeoutError:
                    logger.error(
                        f"Conversion timeout for {file_path} after {CONVERSION_TIMEOUT} seconds"
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

        # Save final statistics
        self.stats.save_stats()

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
            return

        time.sleep(0.1)  # Brief delay to let the file system stabilize

        file_path = event.src_path
        logger.info(f"Detected new file: {file_path}")
        self.queue.put(file_path)


class DocumentWatcher:
    """Watches a directory for document files and processes them as they arrive."""

    def __init__(
        self,
        converter: DocumentConverter,
        input_dir: str,
        output_dir: str,
        stats_file: Optional[str] = None,
    ):
        """Initialize the document watcher.

        Args:
            converter: The converter to use for document processing
            input_dir: The input directory to watch for new documents
            output_dir: The output directory to save converted documents
            stats_file: Optional path to save conversion statistics

        Raises:
            ValueError: If input_dir does not exist or is not a directory
        """
        # Validate input directory
        input_path = Path(input_dir)
        if not input_path.exists():
            raise ValueError(f"Input directory does not exist: {input_dir}")
        if not input_path.is_dir():
            raise ValueError(f"Input path is not a directory: {input_dir}")

        # Ensure output directory exists
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        self.converter = converter
        self.input_dir = str(input_path)
        self.output_dir = str(output_path)
        self.observer = Observer()
        self.event_handler = DocumentEventHandler(
            self.converter, self.output_dir, stats_file
        )
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

        # Get all files in the input directory
        input_files = [f for f in input_path.glob("*") if f.is_file()]

        if not input_files:
            logger.info("No existing files found in input directory")
            return

        logger.info(f"Found {len(input_files)} existing files in input directory")

        # Check which files need processing
        files_to_process = []
        for input_file in input_files:
            # Check if output file already exists
            output_file = output_path / f"{input_file.stem}.md"

            if process_all or not output_file.exists():
                files_to_process.append(str(input_file))

        # Queue the files for processing
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
            # Start the event handler first
            self.event_handler.start_processing()

            # Process existing files if requested
            if process_existing:
                self.process_existing_files()

            # Start watching for new files
            self.observer.schedule(
                self.event_handler, path=self.input_dir, recursive=False
            )
            self.observer.start()

            self.is_running = True
            logger.info(f"Started watching directory: {self.input_dir}")
            logger.info(f"Converted documents will be saved to: {self.output_dir}")
        except Exception as e:
            logger.error(f"Failed to start document watcher: {e}", exc_info=True)
            # Attempt to clean up if start fails
            self.stop()
            raise

    def stop(self) -> None:
        """Stop watching the input directory and queue processing."""
        if not self.is_running:
            logger.debug("DocumentWatcher is not running, nothing to stop")
            return

        logger.info(f"Stopping document watcher for {self.input_dir}")

        # Stop event handler first
        try:
            self.event_handler.stop()
        except Exception as e:
            logger.error(f"Error stopping event handler: {e}", exc_info=True)

        # Then stop the observer
        try:
            self.observer.stop()
            self.observer.join(timeout=5)
            if self.observer.is_alive():
                logger.warning("Observer thread did not terminate cleanly")
        except Exception as e:
            logger.error(f"Error stopping observer: {e}", exc_info=True)

        self.is_running = False
        logger.info(f"Stopped watching {self.input_dir}")
