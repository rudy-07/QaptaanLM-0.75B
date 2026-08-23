"""Schema normalization for KapInstruct-100M instruction datasets.

Normalizes records from multiple diverse schemas (messages, ShareGPT conversations,
instruction-response, problem-solution, query-answer, QA) into a canonical format:

{
    "id": "source_identifier:sample_idx",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ],
    "source": "source_name",
    "domain": "domain_category",
    "language": "en",
    "code_language": "python",
    "has_reasoning": true/false,
    "metadata": {...}
}
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Standard role mapping
ROLE_MAP = {
    "system": "system",
    "sys": "system",
    "user": "user",
    "human": "user",
    "prompter": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "model": "assistant",
}


def detect_code_language_from_text(text: str) -> Optional[str]:
    """Detect programming language from markdown code blocks or keywords."""
    fenced_match = re.search(r"```([a-zA-Z0-9_#+\-]+)\b", text)
    if fenced_match:
        raw_lang = fenced_match.group(1).lower()
        alias_map = {
            "py": "python",
            "python": "python",
            "js": "javascript",
            "javascript": "javascript",
            "ts": "typescript",
            "typescript": "typescript",
            "cpp": "cpp",
            "c++": "cpp",
            "c": "c",
            "cs": "csharp",
            "c#": "csharp",
            "csharp": "csharp",
            "java": "java",
            "go": "go",
            "golang": "go",
            "rs": "rust",
            "rust": "rust",
            "rb": "ruby",
            "ruby": "ruby",
            "php": "php",
            "sql": "sql",
            "sh": "shell",
            "bash": "shell",
            "shell": "shell",
            "zsh": "shell",
            "html": "html",
            "css": "css",
            "dockerfile": "dockerfile",
            "docker": "dockerfile",
            "json": "json",
            "yaml": "yaml",
            "yml": "yaml",
        }
        return alias_map.get(raw_lang, raw_lang)
    return None


def detect_has_reasoning(messages: List[Dict[str, str]]) -> bool:
    """Check whether messages contain explicit reasoning traces."""
    for m in messages:
        c = m.get("content", "")
        if "<think>" in c or "</think>" in c:
            return True
        if "step 1" in c.lower() and "step 2" in c.lower():
            return True
        if "let's think step by step" in c.lower() or "first, let's analyze" in c.lower():
            return True
    return False


def normalize_messages_field(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """Normalize a list of role/content message dicts."""
    normalized = []
    for msg in raw_messages:
        role = msg.get("role") or msg.get("from") or "user"
        content = msg.get("content") or msg.get("value") or ""
        role_norm = ROLE_MAP.get(str(role).lower().strip(), "user")
        content_str = str(content).strip()
        if content_str:
            normalized.append({"role": role_norm, "content": content_str})
    return normalized


def normalize_conversations_field(conversations: List[Dict[str, Any]], system_prompt: Optional[str] = None) -> List[Dict[str, str]]:
    """Normalize ShareGPT-style conversations list [{'from': '...', 'value': '...'}]"""
    messages = []
    if system_prompt and system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
        
    for turn in conversations:
        from_role = str(turn.get("from", "user")).lower().strip()
        content = str(turn.get("value", turn.get("content", ""))).strip()
        role = ROLE_MAP.get(from_role, "user")
        if content:
            messages.append({"role": role, "content": content})
            
    return messages


class InstructNormalizer:
    """Normalizes arbitrary raw records into canonical KapInstruct format."""

    def __init__(self, preserve_reasoning: bool = True):
        self.preserve_reasoning = preserve_reasoning

    def normalize(
        self,
        raw_record: Dict[str, Any],
        source_name: str,
        domain: str,
        format_type: str,
        sample_idx: int = 0,
    ) -> Optional[Dict[str, Any]]:
        """Normalize a raw record based on its configured format type.

        Args:
            raw_record: Raw dict from Hugging Face dataset stream.
            source_name: Unique source identifier.
            domain: Domain / category tag.
            format_type: One of 'messages', 'conversations', 'instruction_response',
                         'problem_solution', 'problem_generated_solution',
                         'query_answer', 'question_answer'.
            sample_idx: Sequential integer index.

        Returns:
            Canonical record dict or None if record is invalid.
        """
        messages: List[Dict[str, str]] = []
        code_lang: Optional[str] = None
        record_id = raw_record.get("id") or f"{source_name}:{sample_idx}"

        # 1. messages format (SmolTalk, Tulu-3)
        if format_type == "messages":
            raw_msgs = raw_record.get("messages", [])
            if isinstance(raw_msgs, list):
                messages = normalize_messages_field(raw_msgs)

        # 2. conversations format (OpenThoughts, OpenHermes)
        elif format_type == "conversations":
            raw_convs = raw_record.get("conversations", [])
            sys_prompt = raw_record.get("system") or raw_record.get("system_prompt")
            if isinstance(raw_convs, list):
                messages = normalize_conversations_field(raw_convs, system_prompt=sys_prompt)

        # 3. instruction_response (Magicoder-Evol, Self-OSS)
        elif format_type == "instruction_response":
            inst = raw_record.get("instruction", "").strip()
            inp = raw_record.get("input", "").strip()
            resp = (raw_record.get("response") or raw_record.get("output") or "").strip()
            
            user_content = f"{inst}\n\n{inp}".strip() if inp else inst
            if user_content and resp:
                messages = [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": resp}
                ]

        # 4. problem_solution (Magicoder-OSS, NuminaMath)
        elif format_type == "problem_solution":
            prob = raw_record.get("problem", "").strip()
            sol = raw_record.get("solution", "").strip()
            if not prob and "messages" in raw_record:
                messages = normalize_messages_field(raw_record["messages"])
            elif prob and sol:
                messages = [
                    {"role": "user", "content": prob},
                    {"role": "assistant", "content": sol}
                ]

        # 5. problem_generated_solution (OpenMathInstruct-2)
        elif format_type == "problem_generated_solution":
            prob = raw_record.get("problem", "").strip()
            gen_sol = (raw_record.get("generated_solution") or raw_record.get("solution") or "").strip()
            exp_ans = raw_record.get("expected_answer")
            
            if prob and gen_sol:
                # Retain generated reasoning and expected answer where available
                assistant_resp = gen_sol
                if exp_ans is not None and str(exp_ans) not in gen_sol:
                    assistant_resp = f"{gen_sol}\n\nFinal Answer: {exp_ans}"
                messages = [
                    {"role": "user", "content": prob},
                    {"role": "assistant", "content": assistant_resp}
                ]

        # 6. query_answer (CodeFeedback)
        elif format_type == "query_answer":
            q = (raw_record.get("query") or raw_record.get("question") or "").strip()
            a = (raw_record.get("answer") or raw_record.get("response") or "").strip()
            if q and a:
                messages = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]

        # 7. question_answer (WebInstructSub, SciQ)
        elif format_type == "question_answer":
            q = (raw_record.get("question") or raw_record.get("orig_question") or "").strip()
            a = (raw_record.get("answer") or raw_record.get("orig_answer") or "").strip()
            if q and a:
                messages = [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a}
                ]

        # Fallback heuristic
        if not messages:
            if "messages" in raw_record:
                messages = normalize_messages_field(raw_record["messages"])
            elif "conversations" in raw_record:
                messages = normalize_conversations_field(raw_record["conversations"])

        # Validate message structure
        if not messages:
            return None

        # Check for user + assistant presence
        has_user = any(m["role"] == "user" for m in messages)
        has_assistant = any(m["role"] == "assistant" for m in messages)
        if not (has_user and has_assistant):
            return None

        # Detect code language from metadata or content
        code_lang = raw_record.get("lang") or raw_record.get("language") or raw_record.get("code_language")
        if not code_lang:
            for m in messages:
                detected = detect_code_language_from_text(m["content"])
                if detected:
                    code_lang = detected
                    break

        has_reasoning = detect_has_reasoning(messages)

        return {
            "id": str(record_id),
            "messages": messages,
            "source": source_name,
            "domain": domain,
            "language": "en",
            "code_language": str(code_lang).lower() if code_lang else None,
            "has_reasoning": has_reasoning,
            "metadata": {
                k: v for k, v in raw_record.items()
                if k not in ("messages", "conversations", "problem", "solution", "generated_solution", "instruction", "output", "response", "query", "answer")
                and isinstance(v, (str, int, float, bool))
            }
        }
