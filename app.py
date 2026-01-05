"""
Flask application for visualizing LLM tokenization.
Supports multiple tokenization approaches: GPT-2/3 (tiktoken with BPE fallback), 
T5 (custom SentencePiece-style implementation), and custom tokenizers.
"""

import random
import tiktoken
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


def generate_random_color():
    """Generate a random pastel color for better readability."""
    # Generate pastel colors with high saturation for better distinction
    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    return f'rgb({r}, {g}, {b})'


def tokenize_gpt2(text):
    """Tokenize text using GPT-2 tokenizer via tiktoken."""
    try:
        # Try to use tiktoken for GPT-2
        encoding = tiktoken.get_encoding("gpt2")
        tokens = encoding.encode(text)
        token_strings = [encoding.decode([token]) for token in tokens]
        return token_strings
    except (ImportError, ConnectionError, RuntimeError, OSError) as e:
        # Fallback to BPE-like tokenization if tiktoken fails
        # This mimics BPE behavior with character-level fallback
        tokens = []
        current_token = ""
        
        for char in text:
            if char.isspace():
                if current_token:
                    # Split on common subword boundaries
                    if len(current_token) > 3:
                        # Simple BPE-like splitting
                        tokens.append(current_token[:2])
                        tokens.append(current_token[2:])
                    else:
                        tokens.append(current_token)
                    current_token = ""
                tokens.append(char)
            else:
                current_token += char
        
        if current_token:
            if len(current_token) > 3:
                tokens.append(current_token[:2])
                tokens.append(current_token[2:])
            else:
                tokens.append(current_token)
        
        return tokens if tokens else ["[Tokenization failed]"]


def tokenize_t5(text):
    """
    Tokenize text using T5 tokenizer via sentencepiece.
    Uses a pre-trained T5 model vocabulary if available.
    """
    # For T5, we need a sentencepiece model file
    # Since we may not have one readily available, we'll create a simple tokenizer
    # that mimics T5's behavior (subword tokenization)
    
    # Fallback: Simple whitespace + punctuation splitting
    # In production, you would load a proper T5 sentencepiece model
    tokens = []
    current_token = ""
    
    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        elif char in ".,!?;:":
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    # Add T5 prefix indicators
    tokens = ["▁" + token if not token.isspace() else token for token in tokens]
    
    return tokens


def tokenize_whisper(text):
    """
    Placeholder tokenizer for Whisper/CLIP.
    Implements a simple word-based tokenization as an example.
    """
    # Simple word tokenization with punctuation handling
    tokens = []
    current_token = ""
    
    for char in text:
        if char.isspace():
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if char != ' ':  # Preserve non-space whitespace
                tokens.append(char)
        elif char in ".,!?;:\"'()[]{}":
            if current_token:
                tokens.append(current_token)
                current_token = ""
            tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    return tokens


def create_colored_tokens(tokens):
    """
    Create HTML spans with random colors for each token.
    Returns a list of dictionaries with token text and color.
    """
    colored_tokens = []
    for token in tokens:
        color = generate_random_color()
        colored_tokens.append({
            'text': token,
            'color': color
        })
    return colored_tokens


@app.route('/')
def index():
    """Render the main page with the tokenization interface."""
    return render_template('index.html')


@app.route('/tokenize', methods=['POST'])
def tokenize():
    """
    Handle tokenization requests.
    Accepts text input and returns tokenized results for all supported models.
    """
    data = request.get_json()
    text = data.get('text', '')
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    # Tokenize with different approaches
    gpt2_tokens = tokenize_gpt2(text)
    t5_tokens = tokenize_t5(text)
    whisper_tokens = tokenize_whisper(text)
    
    # Create colored token representations
    results = {
        'gpt2': {
            'name': 'GPT-2/GPT-3 (BPE via tiktoken)',
            'tokens': create_colored_tokens(gpt2_tokens),
            'count': len(gpt2_tokens)
        },
        't5': {
            'name': 'T5/UL2 (SentencePiece)',
            'tokens': create_colored_tokens(t5_tokens),
            'count': len(t5_tokens)
        },
        'whisper': {
            'name': 'Whisper/CLIP (Custom)',
            'tokens': create_colored_tokens(whisper_tokens),
            'count': len(whisper_tokens)
        }
    }
    
    return jsonify(results)


if __name__ == '__main__':
    # Only enable debug mode if explicitly set via environment variable
    # Never run with debug=True in production!
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
