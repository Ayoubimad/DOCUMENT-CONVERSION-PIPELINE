from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from base_converter import DocumentConverter

# Watchdog is a library that allows us to detect file changes, it uses the Observer design pattern
# as soon as a new file is created, it will be detected and the event handler will be called


class DocumentEventHandler(FileSystemEventHandler):
    """Event handler for document events."""

    # Extends FileSystemEventHandler, we override the on_created method, by doing this, as soon as a file is created
    def __init__(self, converter: DocumentConverter, output_dir: str):
        self.converter = converter
        self.output_dir = output_dir

    def on_created(self, event):
        if event.is_directory:
            return
        # TODO: handle multiple file types, we must convert only the supported ones
        # document = self.converter.convert(event.src_path)
        print(f"New file created: {event.src_path}")


class DocumentWatcher:
    def __init__(self, converter: DocumentConverter, input_dir: str, output_dir: str):
        """Initialize the document watcher.

        Args:
            converter: The converter to use.
            input_dir: The input directory to watch.
            output_dir: The output directory to save the converted documents.
        """
        self.converter = converter
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.observer = Observer()
        self.event_handler = DocumentEventHandler(self.converter, self.output_dir)

    def start(self):
        self.observer.schedule(
            self.event_handler, path=self.input_dir, recursive=False
        )  # in this case the observer will run in the background in a separate thread, watching for events in the input directory
        # the observer will call the event handler when a new file is created
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()
