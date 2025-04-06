# Document Conversion Pipeline

A robust Python application for watching a directory for documents and automatically converting them using the Docling API.

## Overview

This application monitors a specified input directory for new document files, processes them through a document conversion service (Docling), and saves the converted markdown output to the specified output directory.

## Features

- **File Watching**: Automatically detects new files in the watched directory
- **Async Processing**: Uses asynchronous processing for efficient document conversion
- **Stability Detection**: Ensures files are completely written before processing
- **Graceful Shutdown**: Properly handles application shutdown and resource cleanup
- **Error Handling**: Robust error handling throughout the pipeline

## Architecture

The application follows a modular design with clear separation of concerns:

- **Document Model**: Represents document content and metadata
- **HTTP Client**: Handles API communication with proper resource management
- **Document Converter**: Converts documents using the Docling API
- **File Watcher**: Monitors directories for new files and processes them

## Configuration

The application is configured using environment variables that can be set directly or through a `.env` file:

```
DOCLING_HOST=localhost
DOCLING_PORT=8000
DOCLING_TIMEOUT=60.0
INPUT_DIR=/path/to/watch
OUTPUT_DIR=/path/to/save
```

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd document-conversion-pipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration

## Usage

Run the application:

```
python -m src.main
```

The application will:
1. Start watching the input directory
2. Process any new files that appear
3. Save converted markdown files to the output directory

## Development

### Project Structure

```
document-conversion-pipeline/
├── src/
│   ├── __init__.py
│   ├── main.py             # Application entry point
│   ├── config.py           # Configuration handling
│   ├── document.py         # Document model
│   ├── base_client.py      # Base HTTP client
│   ├── docling_client.py   # Docling API client
│   ├── base_converter.py   # Base converter interface
│   ├── docling_converter.py # Docling document converter
│   ├── conversion_option.py # Document conversion options
│   └── watcher.py          # Directory watcher
├── .env                    # Environment configuration
└── README.md               # This file
```

### Extending

To support additional document converters:

1. Create a new converter class that implements `DocumentConverter`
2. Implement all required methods
3. Update `main.py` to use your new converter 