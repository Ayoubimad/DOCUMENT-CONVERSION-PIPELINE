import time

from watcher import DocumentWatcher
from converter import DoclingConverter
from config import settings


def main():
    converter = DoclingConverter()
    watcher = DocumentWatcher(converter, settings.INPUT_DIR, settings.OUTPUT_DIR)
    watcher.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        watcher.stop()


if __name__ == "__main__":
    main()
