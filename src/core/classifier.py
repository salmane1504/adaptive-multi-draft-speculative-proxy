import re
from enum import Enum
from typing import Any, Dict, List


class Domain(str, Enum):
    CODE = "code"
    JSON = "json"
    PROSE = "prose"

class PromptClassifier:
    def __init__(self):
        # Code patterns: Markdown blocks, language keywords, function signatures
        self.code_patterns = re.compile(
            r"(```(python|js|ts|cpp|c|rust|go|java|html|css|sql|sh)?|def\s+\w+|function\s+\w+|class\s+\w+|import\s+\w+|SELECT\s+.*\s+FROM|fn\s+main|\bif\b.*:)",
            re.IGNORECASE
        )
        
        # JSON patterns: Explicit request for JSON structures, schema definitions
        self.json_patterns = re.compile(
            r"(\bjson\b|\bjson_object\b|\{.*?:.*?\}|\"type\"\s*:\s*\"object\")",
            re.IGNORECASE
        )
        
    def classify(self, payload: Dict[str, Any]) -> Domain:
        # 1. Parameter-based override (highest confidence)
        response_format = payload.get("response_format", {})
        if isinstance(response_format, dict) and response_format.get("type") == "json_object":
            return Domain.JSON

        # Extract full conversation context into a single string for pattern scanning
        messages: List[Dict[str, str]] = payload.get("messages", [])
        full_text = " ".join([m.get("content", "") for m in messages])
        
        if not full_text.strip():
            return Domain.PROSE

        # 2. Check System Prompt specifically (gives strong domain hints)
        system_msg = next((m.get("content", "") for m in messages if m.get("role") == "system"), "")
        if "json" in system_msg.lower():
            return Domain.JSON
        if any(kw in system_msg.lower() for kw in ["code", "programmer", "developer", "sql"]):
            return Domain.CODE

        # 3. Text pattern heuristics (Order matters: Code > JSON > Prose)
        if len(self.code_patterns.findall(full_text)) >= 1:
            return Domain.CODE

        if len(self.json_patterns.findall(full_text)) >= 2:
            return Domain.JSON

        return Domain.PROSE