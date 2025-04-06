import os
import queue
import threading
import time
import asyncio
from pathlib import Path
from typing import Optional, Dict, Tuple, Any

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter


# Constants for file stability detection
STABILITY_CHECKS = 3  # Number of consecutive stable checks required
STABILITY_INTERVAL = 0.5  # Time between checks in seconds
MAX_WAIT_TIME = 30  # Maximum time to wait for file stability (seconds)
MIN_GROWTH_RATE = (
    1024  # Minimum growth rate in bytes/second to consider a file still being written
)


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document file system events.

    Detects new or modified files and processes them using the provided document converter.
    """

    def __init__(self, converter: DocumentConverter, output_dir: str):
        """Initialize the document event handler.

        Args:
            converter: The converter to use for document processing
            output_dir: The output directory to save converted documents
        """
        self.converter = converter
        self.output_dir = output_dir
        self.queue: queue.Queue = queue.Queue()
        self.is_running = True
        self.processor_thread: Optional[threading.Thread] = None
        self.file_sizes: Dict[str, Tuple[int, float, int, float]] = {}

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
                print(f"Error processing file: {e}")
                import traceback

                traceback.print_exc()

    def _is_file_stable(self, file_path: str) -> bool:
        """Check if a file has finished being written by monitoring size stability.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file size has stabilized, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist")
                return False

            current_size = os.path.getsize(file_path)
            current_time = time.time()

            if current_size == 0:
                print(f"File {file_path} has zero size, waiting...")
                return False

            # Initialize tracking for new files
            if file_path not in self.file_sizes:
                # Store as (size, timestamp, consecutive_stable_checks, last_size_change_time)
                self.file_sizes[file_path] = (
                    current_size,
                    current_time,
                    0,
                    current_time,
                )
                print(
                    f"Started tracking file {file_path} with size {current_size} bytes"
                )
                return False

            prev_size, prev_time, stable_count, last_change_time = self.file_sizes[
                file_path
            ]

            # Check if waited too long
            if current_time - prev_time > MAX_WAIT_TIME:
                print(
                    f"File {file_path} exceeded maximum wait time of {MAX_WAIT_TIME}s, processing anyway"
                )
                return True

            # Check for size stability
            if current_size == prev_size:
                # Size is stable, increment stable count
                stable_count += 1
                print(
                    f"File {file_path} size stable at {current_size} bytes (check {stable_count}/{STABILITY_CHECKS})"
                )

                # Update tracking data with new timestamp and incremented stable count
                self.file_sizes[file_path] = (
                    current_size,
                    current_time,
                    stable_count,
                    last_change_time,
                )

                # Check if we've reached stability threshold
                if stable_count >= STABILITY_CHECKS:
                    print(
                        f"File {file_path} is fully stable after {stable_count} checks"
                    )
                    return True

                # Not enough consecutive stable checks yet
                time.sleep(STABILITY_INTERVAL)
                return False
            else:
                # Size changed, reset stable count and update last change time
                print(
                    f"File {file_path} size changed from {prev_size} to {current_size} bytes, resetting stability count"
                )

                # If file is growing, check if it's growing very slowly
                # If no change for more than 2 seconds, we might consider it stable
                if current_size > prev_size and (current_time - last_change_time > 2.0):
                    # File is growing very slowly, consider processing it
                    growth_rate = (current_size - prev_size) / (
                        current_time - last_change_time
                    )
                    if growth_rate < MIN_GROWTH_RATE:  # Less than 1KB per second
                        print(
                            f"File {file_path} growing very slowly, considering it stable"
                        )
                        return True

                # Update tracking data with reset stable count and updated change time
                self.file_sizes[file_path] = (
                    current_size,
                    current_time,
                    0,
                    current_time,
                )
                time.sleep(STABILITY_INTERVAL)
                return False

        except (FileNotFoundError, PermissionError) as e:
            print(f"File access error during stability check: {e}")
            return False
        except Exception as e:
            print(f"Error checking file stability: {e}")
            return False

    async def _process_file_async(self, file_path: str) -> None:
        """Process a single file using the converter asynchronously.

        Args:
            file_path: Path to the file to be processed.
        """
        try:
            print(f"Starting async conversion of {file_path}")

            # Use the async version of the converter
            document = await self.converter.convert_async(file_path)

            # Create output path
            file_path_obj = Path(file_path)
            base_name = file_path_obj.stem + ".md"
            output_path = Path(self.output_dir) / base_name

            # Make sure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the content to the output file
            try:
                output_path.write_text(document.content, encoding="utf-8")
                print(f"Successfully converted and saved {file_path} to {output_path}")
            except IOError as e:
                print(f"Error writing output file {output_path}: {e}")
                import traceback

                traceback.print_exc()

        except Exception as e:
            print(f"Failed to convert {file_path}: {e}")
            import traceback

            traceback.print_exc()

    def _run_process_file(self, file_path: str) -> None:
        """Run the async process_file method in the appropriate event loop.

        Args:
            file_path: Path to the file to be processed.
        """
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
                future = asyncio.run_coroutine_threadsafe(
                    self._process_file_async(file_path), loop
                )
                try:
                    # Wait for the result with a timeout
                    future.result(timeout=60)
                except Exception as e:
                    print(f"Error in async processing of {file_path}: {e}")
                    import traceback

                    traceback.print_exc()
            else:
                # If we're not in an async context, we can use run_until_complete
                try:
                    loop.run_until_complete(self._process_file_async(file_path))
                except Exception as e:
                    print(f"Error in processing {file_path}: {e}")
                    import traceback

                    traceback.print_exc()
        except Exception as e:
            print(f"Error setting up async environment for {file_path}: {e}")
            import traceback

            traceback.print_exc()

    def start_processing(self) -> None:
        """Start the queue processor in a separate thread."""
        self.processor_thread = threading.Thread(target=self.process_queue)
        self.processor_thread.daemon = True
        self.processor_thread.start()

    def stop(self) -> None:
        """Stop the queue processor."""
        self.is_running = False
        self.file_sizes.clear()
        if self.processor_thread:
            self.processor_thread.join(timeout=5)

    def on_created(self, event: Any) -> None:
        """Handle file creation events.

        Args:
            event: The file system event
        """
        if event.is_directory:
            return

        time.sleep(0.1)  # Brief delay to let the file system stabilize

        file_path = event.src_path
        print(f"Detected new file: {file_path}")
        self.queue.put(file_path)


class DocumentWatcher:
    """Watches a directory for document files and processes them as they arrive."""

    def __init__(self, converter: DocumentConverter, input_dir: str, output_dir: str):
        """Initialize the document watcher.

        Args:
            converter: The converter to use for document processing
            input_dir: The input directory to watch for new documents
            output_dir: The output directory to save converted documents
        """
        self.converter = converter
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.observer = Observer()
        self.event_handler = DocumentEventHandler(self.converter, self.output_dir)

    def start(self) -> None:
        """Start watching the input directory and processing queue."""
        self.observer.schedule(self.event_handler, path=self.input_dir, recursive=False)
        self.observer.start()
        self.event_handler.start_processing()  # Start queue processor in a thread
        print(f"Started watching {self.input_dir}")

    def stop(self) -> None:
        """Stop watching the input directory and queue processing."""
        self.event_handler.stop()  # Stop queue processor
        self.observer.stop()
        self.observer.join()
        print(f"Stopped watching {self.input_dir}")
