"""
Main application module for document conversion pipeline. Manages application lifecycle,
command-line arguments, configuration, logging setup, and coordinates the document
watcher and converter components.
"""

import asyncio
import argparse
import signal
import sys
import os
import logging
from typing import Optional
from pathlib import Path

from docling_converter import DoclingConverter
from watcher import DocumentWatcher
from config import settings
from logging_utils import LoggingConfig, configure_root_logger

logger = logging.getLogger("document_conversion")


class Application:
    """Main application class that manages the document conversion pipeline."""

    def __init__(self, process_existing: bool = False):
        """Initialize the application components.

        Args:
            process_existing: Whether to process existing files in the input directory
        """
        self.converter = DoclingConverter()
        self.process_existing = process_existing
        self.running = False
        self.event_loop: Optional[asyncio.AbstractEventLoop] = None

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.watcher = DocumentWatcher(
            self.converter, settings.INPUT_DIR, settings.OUTPUT_DIR
        )

    def _signal_handler(self, sig, frame):
        """Handle signals to gracefully shut down the application."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()

    async def start_async(self) -> None:
        """Start the application asynchronously."""
        self.running = True
        logger.info("Starting document conversion pipeline")

        self.event_loop = asyncio.get_running_loop()

        try:
            self.watcher.start(process_existing=self.process_existing)
            logger.info(f"Watching input directory: {settings.INPUT_DIR}")
            logger.info(f"Saving output to: {settings.OUTPUT_DIR}")

            while self.running:
                await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error in application: {e}", exc_info=True)
            self.stop()

    def start(self) -> None:
        """Start the document watcher and register signal handlers."""
        asyncio.run(self.start_async())

    async def cleanup_async(self) -> None:
        """Clean up any async resources."""
        logger.info("Cleaning up resources...")
        try:
            await self.converter.close()
            logger.info("Converter resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up converter: {e}", exc_info=True)

    def stop(self) -> None:
        """Stop the application and clean up resources."""
        if not self.running:
            return

        logger.info("Stopping application...")
        self.running = False

        try:
            self.watcher.stop()
            logger.info("Watcher stopped")
        except Exception as e:
            logger.error(f"Error stopping watcher: {e}", exc_info=True)

        if self.event_loop and self.event_loop.is_running():
            asyncio.run_coroutine_threadsafe(self.cleanup_async(), self.event_loop)
        else:
            asyncio.run(self.cleanup_async())

    def run(self) -> None:
        """Run the application until interrupted."""
        try:
            self.start()
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received, shutting down...")
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Document Conversion Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Override the input directory (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override the output directory (default: from config)",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process existing files in the input directory on startup",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level",
    )
    parser.add_argument(
        "--log-file", type=str, help="Log to specified file in addition to console"
    )
    parser.add_argument(
        "--no-timestamps", action="store_true", help="Disable timestamps in log output"
    )

    return parser.parse_args()


def configure_logging_from_args(args):
    """Configure logging based on command line arguments.

    Args:
        args: Parsed command line arguments
    """
    if args.log_level:
        settings.LOG_LEVEL = args.log_level

    if args.log_file:
        settings.LOG_TO_FILE = True
        settings.LOG_FILE_PATH = args.log_file

    if args.no_timestamps:
        settings.INCLUDE_TIMESTAMPS = False

    settings.configure_logging()


def main() -> None:
    """Application entry point."""
    args = parse_args()

    configure_logging_from_args(args)

    if args.input_dir:
        settings.INPUT_DIR = args.input_dir
        logger.debug(f"Input directory overridden to {settings.INPUT_DIR}")
    if args.output_dir:
        settings.OUTPUT_DIR = args.output_dir
        logger.debug(f"Output directory overridden to {settings.OUTPUT_DIR}")

    os.makedirs(settings.INPUT_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    app = Application(process_existing=args.process_existing)
    app.run()


if __name__ == "__main__":
    main()
