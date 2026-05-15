from __future__ import annotations

def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    try:
        import tiktoken
    except ImportError:
        return len(text.split())

    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(text))
