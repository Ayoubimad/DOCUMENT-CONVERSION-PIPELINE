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

        self.watcher = DocumentWatcher(
            self.converter, settings.INPUT_DIR, settings.OUTPUT_DIR
        )
        self.running = False

    def start(self) -> None:
        """Start the document watcher and register signal handlers."""
        self.running = True

        self.watcher.start(process_existing=self.process_existing)

    def stop(self) -> None:
        """Stop the application and clean up resources."""
        if not self.running:
            return

        self.running = False

        asyncio.run(self._cleanup())

        self.watcher.stop()

    async def _cleanup(self) -> None:
        """Clean up any async resources."""
        await self.converter.close()

    def run(self) -> None:
        """Run the application until interrupted."""
        try:
            self.start()
            while self.running:
                try:
                    asyncio.run(asyncio.sleep(0.1))
                except (KeyboardInterrupt, asyncio.CancelledError):
                    break
        except Exception as e:
            logger.error(f"Error in application: {e}", exc_info=True)
        finally:
            self.stop()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Conversion Pipeline")

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

    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()

    if args.input_dir:
        settings.INPUT_DIR = args.input_dir
        logger.debug(f"Input directory overridden to {settings.INPUT_DIR}")
    if args.output_dir:
        settings.OUTPUT_DIR = args.output_dir
        logger.debug(f"Output directory overridden to {settings.OUTPUT_DIR}")

    process_existing = args.process_existing

    app = Application(process_existing=process_existing)

    app.run()


if __name__ == "__main__":
    main()
