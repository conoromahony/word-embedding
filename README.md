# 🔤 LLM Tokenization Visualizer

A Flask web application that provides an interactive interface for visualizing how different Large Language Models (LLMs) tokenize text. Compare tokenization approaches used by GPT-2/3, T5/UL2, and other models side-by-side with color-coded visual feedback.

<img width="1232" height="1064" alt="Screenshot 2026-01-05 at 3 44 33 PM" src="https://github.com/user-attachments/assets/a90122b8-11da-446c-bbb6-5734626a0f50" />


## Features

- **Multiple Tokenization Approaches**: Compare how different models break down text.

- **Visual Feedback**: Each token is color-coded with a random distinct color for easy identification and readability

- **Interactive Interface**: 
  - Simple text input area
  - Real-time tokenization results
  - Token count display for each approach

- **Responsive Design**: Modern, gradient-styled UI with smooth transitions

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/conoromahony/word-embedding.git
cd word-embedding
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Log into Hugging Face:
To run the Llama tokenization, you need to set up your access token on Hugging Face and then specify it using an environment variable. You also need to have gotten approval from Meta (up on Hugging Face) to use the Llama model.
```bash
export HF_TOKEN=<your_access_token>
```

### Optional: Enable Debug Mode

For development purposes, you can enable Flask's debug mode:
```bash
export FLASK_DEBUG=true
```

**Warning:** Never run with debug mode enabled in production environments as it can expose security vulnerabilities.

## Usage

### Running the Application

Start the Flask development server:
```bash
python app.py
```

The application will be available at `http://127.0.0.1:5001`

### Using the Interface

1. **Enter Text**: Type or paste text into the input area
3. **Tokenize**: Click the "Tokenize Text" button to see the results
4. **Compare**: View how each tokenization approach handles the same text differently

## Project Structure

```
word-embedding/
├── app.py                 # Flask application with tokenization logic
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend interface
└── README.md             # This file
```

## Dependencies

- **Flask** (3.0.0): Web framework
- **Hugging Face**: An access token so the code can access some of the tokenizers on Hugging Face
- **Llama**: Approval from Meta to use its models up on Hugging Face
- **tiktoken** (0.5.2): OpenAI's tokenizer for GPT models
- **sentencepiece** (0.1.99): Google's tokenization library (optional, not currently used)
- **Werkzeug** (3.0.3): WSGI utilities for Flask (security-patched version)

## How It Works

### Color Coding

Each token is assigned a random pastel color from the RGB spectrum to make individual tokens visually distinct and easier to track across the text.

## Development

### Running in Debug Mode (Development Only)

By default, the application runs in production mode for security. To enable debug mode for development:

```bash
export FLASK_DEBUG=true
python app.py
```

Debug mode enables:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

**Important:** Never enable debug mode in production environments!

### Production Deployment

For production deployment, use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn app:app
```

## Contributing

Contributions are welcome! Feel free to:
- Add new tokenization approaches
- Improve the UI/UX
- Enhance tokenization algorithms

## License

This project is open source and available for educational and experimental purposes.

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- Tokenization powered by [tiktoken](https://github.com/openai/tiktoken), [SentencePiece](https://github.com/google/sentencepiece), and [HuggingFace](https://huggingface.co)
- Inspired by the need to understand how different LLMs process text
