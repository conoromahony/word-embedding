# 🔤 LLM Tokenization Visualizer

A Flask web application that provides an interactive interface for visualizing how different Large Language Models (LLMs) tokenize text. Compare tokenization approaches used by GPT-2/3, T5/UL2, and other models side-by-side with color-coded visual feedback.

![Tokenization Visualizer](https://github.com/user-attachments/assets/d69b244c-bcc7-407f-8ce1-ffcd34824386)

## Features

- **Multiple Tokenization Approaches**: Compare how different models break down text:
  - **GPT-2/GPT-3**: Byte Pair Encoding (BPE) using tiktoken
  - **T5/UL2**: SentencePiece tokenization
  - **Whisper/CLIP**: Custom word-based tokenization (placeholder example)

- **Visual Feedback**: Each token is color-coded with a random distinct color for easy identification and readability

- **Interactive Interface**: 
  - Simple text input area
  - Pre-loaded example texts to try
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

The application will be available at `http://127.0.0.1:5000`

### Using the Interface

1. **Enter Text**: Type or paste text into the input area
2. **Try Examples**: Click any of the example buttons to load sample text
3. **Tokenize**: Click the "Tokenize Text" button to see the results
4. **Compare**: View how each tokenization approach handles the same text differently

### Example Texts Included

- Simple Greeting
- Classic Pangram ("The quick brown fox...")
- Technical Text (LLM-related content)
- Text with Emojis & Special Characters

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
- **tiktoken** (0.5.2): OpenAI's tokenizer for GPT models
- **sentencepiece** (0.1.99): Google's tokenization library
- **Werkzeug** (3.0.1): WSGI utilities for Flask

## How It Works

### Tokenization Approaches

1. **GPT-2/GPT-3 (BPE)**:
   - Uses Byte Pair Encoding to create subword tokens
   - Balances vocabulary size with the ability to handle rare words
   - Falls back to BPE-like splitting if tiktoken cannot download encodings

2. **T5/UL2 (SentencePiece)**:
   - Implements subword tokenization with special markers (▁) for word boundaries
   - Handles whitespace and punctuation as separate tokens
   - Provides a balance between word-level and character-level tokenization

3. **Whisper/CLIP (Custom)**:
   - Word-based tokenization with punctuation separation
   - Simpler approach that treats each word as a token
   - Serves as a placeholder for custom tokenization logic

### Color Coding

Each token is assigned a random pastel color from the RGB spectrum to make individual tokens visually distinct and easier to track across the text.

## Screenshots

### Initial Interface
![Initial Page](https://github.com/user-attachments/assets/fabcfdfa-31b4-4209-b1ba-411f8bc9f2b0)

### Tokenization Results
![Results](https://github.com/user-attachments/assets/d69b244c-bcc7-407f-8ce1-ffcd34824386)

### Technical Text Example
![Technical Example](https://github.com/user-attachments/assets/605d22df-401d-4534-8512-f79bbe4a6bfe)

## Development

### Running in Debug Mode

The application runs in debug mode by default, which enables:
- Auto-reload on code changes
- Detailed error messages
- Interactive debugger

For production deployment, use a production WSGI server like Gunicorn:
```bash
pip install gunicorn
gunicorn app:app
```

## Contributing

Contributions are welcome! Feel free to:
- Add new tokenization approaches
- Improve the UI/UX
- Add more example texts
- Enhance tokenization algorithms

## License

This project is open source and available for educational and experimental purposes.

## Acknowledgments

- Built with [Flask](https://flask.palletsprojects.com/)
- Tokenization powered by [tiktoken](https://github.com/openai/tiktoken) and [SentencePiece](https://github.com/google/sentencepiece)
- Inspired by the need to understand how different LLMs process text
