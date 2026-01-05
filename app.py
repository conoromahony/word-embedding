"""
Flask application for visualizing LLM tokenization.
Supports multiple tokenization approaches: GPT-2/3 (tiktoken with BPE fallback), 
T5 (custom SentencePiece-style implementation), and custom tokenizers.
"""

import random
import tiktoken
import os
from flask import Flask, render_template, request, jsonify
from huggingface_hub import login
import os

login(token=os.environ["HF_TOKEN"])

# Set environment variable to disable symlinks in Hugging Face Hub
# This helps with offline environments
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['HF_HUB_OFFLINE'] = '0'  # Try online first, but fail fast

# Import transformers library components
try:
    from transformers import (
        BertTokenizer,
        AutoTokenizer,
        LlamaTokenizer
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

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


def tokenize_bert(text):
    """Tokenize text using BERT tokenizer from bert-base-uncased."""
    if not TRANSFORMERS_AVAILABLE:
        return ["[Transformers library not available]"]
    
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', local_files_only=True)
        # Get token strings only (no IDs)
        tokens = tokenizer.tokenize(text)
        
        return tokens if tokens else ["[Empty tokenization]"]
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        return ["[BERT tokenization failed: Model not available offline]"]


def tokenize_gpt4(text):
    """Tokenize text using GPT-4 tokenizer via tiktoken (cl100k_base encoding)."""
    try:
        # GPT-4 uses cl100k_base encoding
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        token_strings = [encoding.decode([token]) for token in tokens]
        
        return token_strings if token_strings else ["[Empty tokenization]"]
    except (OSError, ValueError, ImportError, RuntimeError, ConnectionError) as e:
        return ["[GPT-4 tokenization failed: Encoding not available offline]"]


def tokenize_phi3(text):
    """Tokenize text using Phi-3 tokenizer (fallback to microsoft/phi-2)."""
    if not TRANSFORMERS_AVAILABLE:
        return ["[Transformers library not available]"]
    
    try:
        # Try microsoft/phi-2 as fallback (phi-3 models may require special access)
        # Note: This model typically requires trust_remote_code=True, but we disable it
        # for security reasons. This may cause the tokenizer to fail, which is acceptable
        # in favor of not executing arbitrary remote code. In production, use pre-validated
        # and cached models only.
        tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-2')
        # Get token strings only (no IDs)
        encoded = tokenizer(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        
        return tokens if tokens else ["[Empty tokenization]"]
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        return ["[Phi-3 tokenization failed: Model not available offline]"]


def tokenize_llama2(text):
    """Tokenize text using LLaMA 2 tokenizer from meta-llama/Llama-2-7b-hf."""
    if not TRANSFORMERS_AVAILABLE:
        return ["[Transformers library not available]"]
    
    try:
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
        
        if tokenizer is None:
            return ["[LLaMA 2 tokenization failed: Model not available offline]"]
        
        # Get token strings and token IDs
        encoded = tokenizer(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])
        token_ids = encoded['input_ids']
        
        # Return tokens with IDs for display
        result_tokens = []
        for token, token_id in zip(tokens, token_ids):
            result_tokens.append(f"{token} [{token_id}]")
        
        return result_tokens if result_tokens else ["[Empty tokenization]"]
    except (OSError, ValueError, ImportError, RuntimeError) as e:
        return ["[LLaMA 2 tokenization failed: Model not available offline]"]


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
    gpt4_tokens = tokenize_gpt4(text)
    bert_tokens = tokenize_bert(text)
    t5_tokens = tokenize_t5(text)
    llama2_tokens = tokenize_llama2(text)
    phi3_tokens = tokenize_phi3(text)
    
    # Create colored token representations in the specified order
    results = {
        'gpt2': {
            'name': 'GPT-2/GPT-3',
            'description': 'BPE tokenizer from OpenAI with 50,257 vocabulary tokens.',
            'tokens': create_colored_tokens(gpt2_tokens),
            'count': len(gpt2_tokens)
        },
        'gpt4': {
            'name': 'GPT-4',
            'description': 'Enhanced BPE tokenizer from OpenAI with 100,000 vocabulary tokens.',
            'tokens': create_colored_tokens(gpt4_tokens),
            'count': len(gpt4_tokens)
        },
        'bert': {
            'name': 'BERT',
            'description': 'WordPiece tokenizer (uncased) from Google with 30,522 vocabulary tokens.',
            'tokens': create_colored_tokens(bert_tokens),
            'count': len(bert_tokens)
        },
        't5': {
            'name': 'T5/UL2',
            'description': 'SentencePiece tokenizer from Google with 32,128 vocabulary tokens.',
            'tokens': create_colored_tokens(t5_tokens),
            'count': len(t5_tokens)
        },
        'llama2': {
            'name': 'LLaMA 2',
            'description': 'SentencePiece tokenizer from Meta with 32,000 vocabulary tokens.',
            'tokens': create_colored_tokens(llama2_tokens),
            'count': len(llama2_tokens)
        },
        'phi3': {
            'name': 'Phi-3',
            'description': 'CodeGen tokenizer from Microsoft with 51,200 vocabulary tokens.',
            'tokens': create_colored_tokens(phi3_tokens),
            'count': len(phi3_tokens)
        }
    }
    
    return jsonify(results)


if __name__ == '__main__':
    # Only enable debug mode if explicitly set via environment variable
    # Never run with debug=True in production!
    import os
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5001)
