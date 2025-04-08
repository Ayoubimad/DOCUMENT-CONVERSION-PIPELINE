# A simple Document Conversion Pipeline

A Python application for monitoring a directory and automatically converting documents using the Docling API.

## Overview

This application watches a specified input directory for new document files, processes them through the [Docling API](https://github.com/docling-project/docling-serve), and saves the converted markdown output to an output directory.

## Features

- **File Watching**: Automatically detects new files in the watched directory.
- **Stability Detection**: Ensures files are completely written before processing.

## Configuration

Configure the application using environment variables, either set directly or via a `.env` file:

```plaintext
DOCLING_HOST=localhost:5001 
DOCLING_PORT=8000
DOCLING_TIMEOUT=60.0
INPUT_DIR=/path/to/watch
OUTPUT_DIR=/path/to/save
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-conversion-pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your configuration.

## Usage

Run the application:

```bash
cd src
python main.py
```

The application will:
1. Start watching the input directory.
2. Process any new files that appear.
3. Save converted markdown files to the output directory.

## Command-Line Arguments

Configure the application using the following command-line arguments:

- `--input-dir`: Override the input directory (default: from config).
- `--output-dir`: Override the output directory (default: from config).
- `--process-existing`: Process existing files in the input directory on startup.

## Development

### Project Structure

```plaintext
document-conversion-pipeline/
├── src/
│   ├── __init__.py
│   ├── .env                # Environment configuration
│   ├── main.py             # Application entry point
│   ├── config.py           # Configuration handling
│   ├── document.py         # Document model
│   ├── base_client.py      # Base HTTP client
│   ├── docling_client.py   # Docling API client
│   ├── base_converter.py   # Base converter interface
│   ├── docling_converter.py # Docling document converter
│   ├── conversion_option.py # Document conversion options
│   └── watcher.py          # File System Event Handler
└── README.md               # This file
```

### Extending

To support additional document converters:

1. Create a new converter class that implements `DocumentConverter`.
2. Implement all required methods.
3. Update `main.py` to use your new converter.

### TODO
1. docling_converter.py: convert and convert_all should be sync
2. A retry mechanism if a given conversion fails

