import asyncio
import signal
import sys
from typing import Optional

from docling_converter import DoclingConverter
from watcher import DocumentWatcher
from config import settings


class Application:
    """Main application class that manages the document conversion pipeline."""

    def __init__(self):
        """Initialize the application components."""
        self.converter = DoclingConverter()
        self.watcher = DocumentWatcher(
            self.converter, settings.INPUT_DIR, settings.OUTPUT_DIR
        )
        self.running = False

    def start(self) -> None:
        """Start the document watcher and register signal handlers."""
        self.running = True

        self.watcher.start()
        print(f"Document conversion pipeline started. Watching {settings.INPUT_DIR}")
        print(f"Converted documents will be saved to {settings.OUTPUT_DIR}")

    def stop(self) -> None:
        """Stop the application and clean up resources."""
        if not self.running:
            return

        print("Shutting down document conversion pipeline...")
        self.running = False

        asyncio.run(self._cleanup())

        self.watcher.stop()
        print("Document conversion pipeline stopped.")

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
            print(f"Error in application: {e}")
            import traceback

            traceback.print_exc()
        finally:
            self.stop()


def main() -> None:
    """Application entry point."""
    app = Application()
    app.run()


if __name__ == "__main__":
    main()
