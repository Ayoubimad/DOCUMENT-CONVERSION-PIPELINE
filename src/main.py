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

    def __init__(self, stats_enabled: bool = False, process_existing: bool = False):
        """Initialize the application components.

        Args:
            stats_enabled: Whether to enable statistics tracking
            process_existing: Whether to process existing files in the input directory
        """
        self.converter = DoclingConverter()
        self.process_existing = process_existing

        stats_file = None
        if stats_enabled:
            stats_dir = Path(settings.OUTPUT_DIR) / "stats"
            stats_dir.mkdir(exist_ok=True, parents=True)
            stats_file = str(stats_dir / "conversion_stats.json")

        self.watcher = DocumentWatcher(
            self.converter, settings.INPUT_DIR, settings.OUTPUT_DIR, stats_file
        )
        self.running = False

    def start(self) -> None:
        """Start the document watcher and register signal handlers."""
        self.running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.watcher.start(process_existing=self.process_existing)
        logger.info(
            f"Document conversion pipeline started. Watching {settings.INPUT_DIR}"
        )
        logger.info(f"Converted documents will be saved to {settings.OUTPUT_DIR}")

    def stop(self) -> None:
        """Stop the application and clean up resources."""
        if not self.running:
            return

        logger.info("Shutting down document conversion pipeline...")
        self.running = False

        asyncio.run(self._cleanup())

        self.watcher.stop()
        logger.info("Document conversion pipeline stopped.")

    async def _cleanup(self) -> None:
        """Clean up any async resources."""
        await self.converter.close()

    def _signal_handler(self, sig: int, frame: Optional[object]) -> None:
        """Handle OS signals for graceful shutdown.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        logger.info(f"Received signal {sig}, shutting down...")
        self.stop()
        sys.exit(0)

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


def setup_logging(debug_mode: bool = False) -> None:
    """Set up application logging.

    Args:
        debug_mode: Enable debug logging if True
    """
    log_level = logging.DEBUG if debug_mode else logging.INFO

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
        ],
    )

    if not debug_mode:
        logging.getLogger("watchdog").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

    if debug_mode:
        logger.debug("Debug logging enabled")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Document Conversion Pipeline")
    parser.add_argument(
        "--stats", action="store_true", help="Enable statistics tracking"
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
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="Process existing files in the input directory on startup",
    )
    parser.add_argument(
        "--process-all",
        action="store_true",
        help="Process all existing files, even if output files already exist (implies --process-existing)",
    )

    return parser.parse_args()


def main() -> None:
    """Application entry point."""
    args = parse_args()

    setup_logging(debug_mode=args.debug)

    if args.input_dir:
        settings.INPUT_DIR = args.input_dir
        logger.debug(f"Input directory overridden to {settings.INPUT_DIR}")
    if args.output_dir:
        settings.OUTPUT_DIR = args.output_dir
        logger.debug(f"Output directory overridden to {settings.OUTPUT_DIR}")

    os.makedirs(settings.INPUT_DIR, exist_ok=True)
    os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

    process_existing = args.process_existing or args.process_all

    app = Application(stats_enabled=args.stats, process_existing=process_existing)

    if args.process_all and process_existing:
        logger.info("Processing all existing files, even if output files exist")
        app.watcher.process_existing_files(process_all=True)

    app.run()


if __name__ == "__main__":
    main()
