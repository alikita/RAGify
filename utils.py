"""
utils.py - Utility Functions and Helpers

This module provides helper utilities for:
1. Formatting file size byte outputs into human-readable strings.
2. Validating file extensions and types.
3. Formatting and exporting chat history logs into downloadable text files.
4. General error logging helper.
"""

import os
from typing import List, Dict, Tuple, Any

def format_bytes(size_bytes: int) -> str:
    """
    Converts bytes into a human-readable format (B, KB, MB, GB).
    """
    if size_bytes == 0:
        return "0 B"
    
    size_name = ("B", "KB", "MB", "GB")
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def validate_file(filename: str, size_bytes: int, max_mb: int = 15) -> Tuple[bool, str]:
    """
    Validates the uploaded file extension and size.
    Returns:
        tuple (is_valid, error_message)
    """
    allowed_extensions = {".pdf", ".txt"}
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return False, f"Unsupported file type '{file_ext}'. RAGify only supports PDF and TXT documents."
        
    max_bytes = max_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return False, f"File exceeds size limit of {max_mb}MB. Please upload a smaller file."
        
    return True, ""

def generate_history_download(chat_history: List[Dict[str, str]]) -> str:
    """
    Compiles chat history logs into a beautifully structured plain text file,
    perfect for exports.
    """
    if not chat_history:
        return "No conversation history recorded."
        
    export_lines = []
    export_lines.append("=========================================================================")
    export_lines.append("                    RAGify - Conversation History Export                  ")
    export_lines.append(f"                    Date & Time: {time_str()}")
    export_lines.append("=========================================================================\n")
    
    for i, msg in enumerate(chat_history, 1):
        role = msg["role"].upper()
        content = msg["content"]
        export_lines.append(f"[{role} - Turn {i}]")
        export_lines.append(content)
        export_lines.append("-" * 73 + "\n")
        
    return "\n".join(export_lines)

def time_str() -> str:
    """Returns local date-time string."""
    import datetime
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def extract_meta_statistics(file_details: List[Dict[str, Any]], num_chunks: int) -> Dict[str, Any]:
    """
    Aggregates files metadata statistics for UI presentation.
    """
    total_size = sum(f["size"] for f in file_details)
    total_pages = sum(f["pages"] for f in file_details)
    
    return {
        "total_files": len(file_details),
        "total_size_friendly": format_bytes(total_size),
        "total_pages": total_pages,
        "total_chunks": num_chunks
    }
