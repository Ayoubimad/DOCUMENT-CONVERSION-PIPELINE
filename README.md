# Document Conversion Pipeline

A Python application for monitoring a directory and automatically converting documents using the Docling API.

## Overview

This application watches a specified input directory for new document files, processes them through the [Docling API](https://github.com/docling-project/docling-serve), and saves the converted markdown output to an output directory.

## Features

- **Asynchronous Processing**: Efficiently processes documents using asyncio.
- **File Watching**: Automatically detects new files in the watched directory.
- **Stability Detection**: Ensures files are completely written before processing.
- **Retry Mechanism**: Automatically retries failed conversions with exponential backoff.
- **Concurrency Control**: Limits the number of simultaneous conversions to prevent overwhelming the API.
- **Colored Logging**: Visual differentiation of log levels for improved readability.
- **Flexible Logging**: Support for file-based and console logging with customizable formats.

## Configuration

Configure the application using environment variables, either set directly or via a `.env` file:

```plaintext
# Required settings
DOCLING_HOST=localhost
DOCLING_PORT=8000
INPUT_DIR=/path/to/watch
OUTPUT_DIR=/path/to/save

# Optional settings with defaults
DOCLING_TIMEOUT=300.0
DOCLING_SSL=False
MAX_RETRIES=3
RETRY_DELAY=2.0
MAX_RETRY_DELAY=60.0
MAX_CONCURRENT_CONVERSIONS=3
STABILITY_CHECKS=3
STABILITY_INTERVAL=0.5

# Logging settings
LOG_LEVEL=INFO
LOG_TO_FILE=False
LOG_FILE_PATH=
INCLUDE_TIMESTAMPS=True
```

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd document-conversion-pipeline
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your configuration.

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
- `--log-level`: Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
- `--log-file`: Log to specified file in addition to console.
- `--no-timestamps`: Disable timestamps in log output.

Examples:
```bash
# Run with custom input/output directories and process existing files
python main.py --input-dir /custom/input --output-dir /custom/output --process-existing

# Run with DEBUG logging to both console and file
python main.py --log-level DEBUG --log-file /path/to/logs/app.log

# Run with minimal logging (no timestamps)
python main.py --no-timestamps
```

## Error Handling

The application includes several error handling mechanisms:

1. **File Stability Detection**: Files are only processed after they have stopped changing size, ensuring they're completely written.
2. **Conversion Retries**: Failed conversions are automatically retried with exponential backoff.
3. **Concurrency Control**: The number of simultaneous conversions is limited to prevent overwhelming the API.
4. **Timeout Scaling**: Larger files are given proportionally more time for conversion.

## Logging

The application uses a comprehensive logging system with the following features:

- **Colored Console Output**: Log levels are color-coded for easy recognition:
  - DEBUG: Cyan
  - INFO: Green
  - WARNING: Yellow
  - ERROR: Red
  - CRITICAL: Red with white background

- **File Logging**: Optionally log to a file in addition to the console.

- **Customizable Format**: Control what information appears in the logs.

Configure logging via environment variables or command-line arguments.

## Development

### Project Structure

```plaintext
document-conversion-pipeline/
├── src/
│   ├── main.py             # Application entry point
│   ├── config.py           # Configuration handling
│   ├── document.py         # Document model
│   ├── base_client.py      # Base HTTP client
│   ├── docling_client.py   # Docling API client
│   ├── base_converter.py   # Base converter interface
│   ├── docling_converter.py # Docling document converter
│   ├── conversion_option.py # Document conversion options
│   ├── watcher.py          # File System Event Handler
│   ├── logging_utils.py    # Logging utilities
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .env                    # Environment configuration (create this)
```

### Extending

To support additional document converters:

1. Create a new converter class that implements `DocumentConverter`.
2. Implement all required methods.
3. Update `main.py` to use your new converter.

### Performance Tuning

The application has several parameters that can be tuned for performance:

- `MAX_CONCURRENT_CONVERSIONS`: Limit the number of simultaneous conversions.
- `STABILITY_CHECKS`: Number of times a file must be checked without changing before being considered stable.
- `STABILITY_INTERVAL`: Time between stability checks.
- `DOCLING_TIMEOUT`: Base timeout for API calls.

## Troubleshooting

### Common Issues

- **File not being processed**: Check if the file is stable and that the input directory is being watched.
- **Timeouts during conversion**: Increase the `DOCLING_TIMEOUT` setting.
- **High CPU usage**: Reduce the `MAX_CONCURRENT_CONVERSIONS` setting.

