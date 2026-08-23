"""Domain-specific and quality filters for KapInstruct-100M instruction datasets.

Includes:
- English language detection (FastText + robust heuristic fallback)
- Programming language normalization and secret scanning
- Math formatting validation (LaTeX balance, OCR artifact detection)
- General instruction checks (min/max length, repetition, prompt injection detection)
- FilterStats tracking per-source rejection metrics
"""

import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# Leaked secrets regex patterns
SECRET_PATTERNS = [
    re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS Access Key
    re.compile(r"ghp_[0-9a-zA-Z]{36}"),  # GitHub Personal Access Token
    re.compile(r"gho_[0-9a-zA-Z]{36}"),  # GitHub OAuth Token
    re.compile(r"glpat-[0-9a-zA-Z\-_]{20}"),  # GitLab Personal Access Token
    re.compile(r"sk-[a-zA-Z0-9]{32,64}"),  # OpenAI API Key
    re.compile(r"xox[baprs]-[0-9a-zA-Z]{10,48}"),  # Slack Token
    re.compile(r"eyJ[a-zA-Z0-9_\-]+\.eyJ[a-zA-Z0-9_\-]+\.[a-zA-Z0-9_\-]+"),  # JWT Token
]

# Prompt injection artifacts
INJECTION_PATTERNS = [
    re.compile(r"ignore previous instructions", re.IGNORECASE),
    re.compile(r"disregard all previous prompts", re.IGNORECASE),
    re.compile(r"<\|system\|>\s*you are a helpful assistant", re.IGNORECASE),
]

# OCR Artifact pattern
OCR_ARTIFACTS = re.compile(
    r"[^\x00-\x7F]{12,}|"  # Long uninterrupted non-ASCII
    r"[\x00-\x08\x0e-\x1f]{3,}|"  # Control characters
    r"(.)\1{12,}|"  # Repeated single character
    r"[|]{6,}"  # Repeated pipes
)

# STEM keywords for domain filtering
STEM_KEYWORDS = {
    "physics", "chemistry", "biology", "science", "math", "mathematics",
    "engineering", "computer", "algorithm", "mechanics", "astronomy",
    "geology", "thermodynamics", "quantum", "circuit", "biochemistry",
    "calculus", "algebra", "geometry", "electromagnetism", "optics",
    "molecule", "reaction", "genetics", "cellular", "differential",
    "matrix", "velocity", "acceleration", "force", "energy", "gravity",
    "equation", "theorem", "hypothesis", "electron", "proton", "neutron"
}


class FilterStats:
    """Track filtering statistics per dataset source."""

    def __init__(self):
        self.total_seen = 0
        self.total_passed = 0
        self.reasons: Dict[str, int] = {}

    def record_pass(self):
        self.total_seen += 1
        self.total_passed += 1

    def record_reject(self, reason: str):
        self.total_seen += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1

    @property
    def pass_rate(self) -> float:
        return self.total_passed / max(self.total_seen, 1)

    def summary(self) -> Dict[str, Any]:
        return {
            "total_seen": self.total_seen,
            "total_passed": self.total_passed,
            "pass_rate": f"{self.pass_rate:.2%}",
            "rejection_reasons": dict(sorted(self.reasons.items(), key=lambda x: -x[1])),
        }


class LanguageDetector:
    """Language detector using FastText with fallback English heuristics."""

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self._load_fasttext(model_path)

    def _load_fasttext(self, model_path: Optional[str]):
        try:
            import fasttext
            from pathlib import Path
            candidates = [
                model_path,
                "lid.176.bin",
                "/tmp/lid.176.bin",
                str(Path.home() / ".cache" / "fasttext" / "lid.176.bin"),
            ]
            for p in candidates:
                if p and Path(p).exists():
                    self.model = fasttext.load_model(str(p))
                    break
        except Exception:
            pass

    def detect(self, text: str) -> Tuple[str, float]:
        if not text or len(text.strip()) < 15:
            return ("en", 0.90)

        if self.model is not None:
            clean = text.replace("\n", " ")[:3000]
            pred = self.model.predict(clean, k=1)
            lang = pred[0][0].replace("__label__", "")
            conf = float(pred[1][0])
            return (lang, conf)

        return self._heuristic_detect(text)

    def _heuristic_detect(self, text: str) -> Tuple[str, float]:
        ascii_chars = sum(1 for c in text if ord(c) < 128)
        ascii_ratio = ascii_chars / max(len(text), 1)
        if ascii_ratio < 0.50:
            return ("unknown", 0.1)

        words = set(re.findall(r"\b[a-z]{2,10}\b", text[:2000].lower()))
        common_en = {
            "the", "and", "is", "of", "to", "in", "that", "with", "for", "as",
            "this", "are", "from", "at", "by", "an", "be", "have", "not", "or",
            "it", "you", "which", "on", "we", "can", "if", "code", "file", "function",
            "return", "given", "following", "calculate", "solve", "determine"
        }
        overlap = len(words.intersection(common_en))
        if ascii_ratio > 0.85 and overlap >= 2:
            return ("en", 0.95)
        elif ascii_ratio > 0.70 and overlap >= 1:
            return ("en", 0.80)
        return ("en", 0.70) if ascii_ratio > 0.80 else ("unknown", 0.30)


def contains_leaked_secrets(text: str) -> bool:
    """Check if text contains API keys or private credentials."""
    for pattern in SECRET_PATTERNS:
        if pattern.search(text):
            return True
    return False


def has_broken_latex(text: str) -> bool:
    """Check for unmatched LaTeX math tags."""
    # Unmatched double dollar signs
    double_dollars = text.count("$$")
    if double_dollars % 2 != 0:
        return True

    # Unmatched begin/end environments
    begins = text.count(r"\begin{")
    ends = text.count(r"\end{")
    if begins > 0 and abs(begins - ends) > 2:
        return True

    return False


def is_stem_content(text: str) -> bool:
    """Check if natural language text relates to STEM domains."""
    words = set(re.findall(r"\b[a-z]{3,15}\b", text.lower()))
    return bool(words.intersection(STEM_KEYWORDS))


def filter_instruction_sample(
    sample: Dict[str, Any],
    config: Dict[str, Any],
    lang_detector: Optional[LanguageDetector] = None,
    stats: Optional[FilterStats] = None,
    source_cfg: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Apply quality and domain filters to a canonical KapInstruct record.

    Args:
        sample: Canonical record dict.
        config: Filter section from config YAML.
        lang_detector: LanguageDetector instance.
        stats: FilterStats tracker.
        source_cfg: Specific source configuration dict.

    Returns:
        The sample dict if it passes all filters, None otherwise.
    """
    messages = sample.get("messages", [])
    if not messages:
        if stats: stats.record_reject("no_messages")
        return None

    # Concatenate all user text and assistant text
    user_text = "\n".join(m["content"] for m in messages if m["role"] == "user")
    asst_text = "\n".join(m["content"] for m in messages if m["role"] == "assistant")
    full_text = f"{user_text}\n{asst_text}"

    gen_cfg = config.get("general", {})
    min_prompt_len = gen_cfg.get("min_prompt_length_chars", 10)
    min_resp_len = gen_cfg.get("min_response_length_chars", 15)
    max_resp_len = gen_cfg.get("max_response_length_chars", 120_000)

    # 1. Length validation
    if len(user_text) < min_prompt_len:
        if stats: stats.record_reject("prompt_too_short")
        return None
    if len(asst_text) < min_resp_len:
        if stats: stats.record_reject("response_too_short")
        return None
    if len(asst_text) > max_resp_len:
        if stats: stats.record_reject("response_too_long")
        return None

    # 2. Leaked Secrets Scan
    code_cfg = config.get("code_languages", {})
    if code_cfg.get("reject_secrets", True):
        if contains_leaked_secrets(full_text):
            if stats: stats.record_reject("contains_leaked_secret")
            return None

    # 3. Prompt Injection Scan
    if gen_cfg.get("reject_prompt_injection", True):
        for pat in INJECTION_PATTERNS:
            if pat.search(user_text):
                if stats: stats.record_reject("prompt_injection")
                return None

    # 4. OCR / Noise Artifacts
    math_cfg = config.get("math", {})
    if math_cfg.get("reject_ocr_artifacts", True):
        if OCR_ARTIFACTS.search(full_text):
            if stats: stats.record_reject("ocr_noise_artifact")
            return None

    # 5. LaTeX Balance (for math/STEM records)
    if math_cfg.get("reject_broken_latex", True) and ("$" in full_text or r"\begin" in full_text):
        if has_broken_latex(full_text):
            if stats: stats.record_reject("broken_latex")
            return None

    # 6. Repetition check
    max_rep_ratio = gen_cfg.get("max_repeated_line_ratio", 0.35)
    lines = [l.strip() for l in asst_text.split("\n") if len(l.strip()) > 10]
    if len(lines) > 6:
        from collections import Counter
        counts = Counter(lines)
        repeated = sum(c - 1 for c in counts.values() if c > 1)
        if (repeated / len(lines)) > max_rep_ratio:
            if stats: stats.record_reject("excessive_repetition")
            return None

    # 7. English Language Filtering
    lang_cfg = config.get("language_detection", {})
    if lang_cfg.get("enabled", True) and lang_detector:
        min_conf = lang_cfg.get("min_english_confidence", 0.65)
        # Check user prompt language (or assistant text if user prompt is minimal code)
        check_text = user_text if len(user_text) > 40 else asst_text
        lang, conf = lang_detector.detect(check_text)
        if lang != "en" and conf >= min_conf:
            if stats: stats.record_reject("non_english")
            return None

    # 8. STEM domain filter (if explicitly required for subset, e.g. WebInstructSub)
    if source_cfg and source_cfg.get("filter_to_stem", False):
        if not is_stem_content(full_text):
            if stats: stats.record_reject("non_stem_domain")
            return None

    if stats:
        stats.record_pass()

    return sample
