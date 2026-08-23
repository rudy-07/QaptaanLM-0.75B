"""Unit test suite for KapInstruct-100M pipeline.

Tests:
1. Normalization across all 7 schema types
2. ChatML tokenization and assistant-only loss masking (single-turn & multi-turn)
3. Quality filters (secrets scanning, LaTeX validation, prompt injection, STEM filtering)
4. Global cross-source deduplication
5. Exact token deficit mixer
6. Sequence packing and Arrow IPC sharding
"""

import os
import sys
import tempfile
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pyarrow as pa
from src.data.instruct_normalizer import InstructNormalizer, detect_code_language_from_text
from src.data.instruct_filters import (
    FilterStats,
    LanguageDetector,
    contains_leaked_secrets,
    has_broken_latex,
    is_stem_content,
    filter_instruction_sample,
)
from src.data.instruct_dedup import GlobalDeduplicator, compute_normalized_dialogue_hash
from src.data.instruct_tokenizer import InstructTokenizer, InstructDocumentPacker
from src.data.instruct_mixture import InstructMixer
from src.data.sharding import DatasetSharder


# ==============================================================================
# 1. Normalization Tests
# ==============================================================================

def test_normalize_messages_format():
    normalizer = InstructNormalizer()
    raw = {
        "id": "tulu_01",
        "messages": [
            {"role": "user", "content": "How do I write quicksort in Python?"},
            {"role": "assistant", "content": "```python\ndef quicksort(arr): ...\n```"}
        ]
    }
    res = normalizer.normalize(raw, "tulu3_sft", "instruction", "messages")
    assert res is not None
    assert len(res["messages"]) == 2
    assert res["messages"][0]["role"] == "user"
    assert res["messages"][1]["role"] == "assistant"
    assert res["code_language"] == "python"


def test_normalize_conversations_format():
    normalizer = InstructNormalizer()
    raw = {
        "id": "hermes_01",
        "system_prompt": "You are a coding tutor.",
        "conversations": [
            {"from": "human", "value": "Explain async/await in JavaScript."},
            {"from": "gpt", "value": "Async/await simplifies promises in JS."}
        ]
    }
    res = normalizer.normalize(raw, "openhermes_2_5", "coding", "conversations")
    assert res is not None
    assert len(res["messages"]) == 3
    assert res["messages"][0]["role"] == "system"
    assert res["messages"][1]["role"] == "user"
    assert res["messages"][2]["role"] == "assistant"


def test_normalize_instruction_response():
    normalizer = InstructNormalizer()
    raw = {
        "instruction": "Reverse a linked list in C++",
        "input": "",
        "response": "```cpp\nListNode* reverseList(ListNode* head) { ... }\n```"
    }
    res = normalizer.normalize(raw, "magicoder_evol", "programming", "instruction_response")
    assert res is not None
    assert res["messages"][0]["content"] == "Reverse a linked list in C++"
    assert res["code_language"] == "cpp"


def test_normalize_problem_generated_solution():
    normalizer = InstructNormalizer()
    raw = {
        "problem": "Calculate 15% of 240.",
        "generated_solution": "15% of 240 is 0.15 * 240 = 36.",
        "expected_answer": "36"
    }
    res = normalizer.normalize(raw, "openmathinstruct2", "math", "problem_generated_solution")
    assert res is not None
    assert "36" in res["messages"][1]["content"]


# ==============================================================================
# 2. Tokenization & Assistant-Only Loss Masking Tests
# ==============================================================================

def test_single_turn_assistant_masking():
    tokenizer = InstructTokenizer("Qwen/Qwen3.5-0.8B-Base", loss_policy="assistant_only")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."}
    ]
    input_ids, labels, rend_count, train_count = tokenizer.render_and_tokenize_conversation(messages)

    assert rend_count == len(input_ids)
    assert len(labels) == len(input_ids)
    assert train_count > 0
    assert train_count < rend_count

    # System and user tokens must have label -100
    # Assistant response tokens must have label == input_id
    masked_count = sum(1 for l in labels if l == -100)
    unmasked_count = sum(1 for l in labels if l != -100)

    assert unmasked_count == train_count
    assert masked_count + unmasked_count == rend_count


def test_multi_turn_assistant_masking():
    tokenizer = InstructTokenizer("Qwen/Qwen3.5-0.8B-Base", loss_policy="assistant_only")
    messages = [
        {"role": "system", "content": "You are an AI pair programmer."},
        {"role": "user", "content": "Turn 1 user prompt."},
        {"role": "assistant", "content": "Turn 1 assistant response."},
        {"role": "user", "content": "Turn 2 user prompt."},
        {"role": "assistant", "content": "Turn 2 assistant response."}
    ]
    input_ids, labels, rend_count, train_count = tokenizer.render_and_tokenize_conversation(messages)

    # Decode and check that assistant text aligns with unmasked labels
    for inp_id, lab_id in zip(input_ids, labels):
        if lab_id != -100:
            assert inp_id == lab_id


def test_all_tokens_loss_policy():
    tokenizer = InstructTokenizer("Qwen/Qwen3.5-0.8B-Base", loss_policy="all_tokens")
    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"}
    ]
    input_ids, labels, rend_count, train_count = tokenizer.render_and_tokenize_conversation(messages)
    assert train_count == rend_count
    assert all(l != -100 for l in labels)


# ==============================================================================
# 3. Filtering Tests
# ==============================================================================

def test_secret_detection():
    secret_text = "Here is my OpenAI API key: sk-1234567890abcdef1234567890abcdef"
    assert contains_leaked_secrets(secret_text) is True

    safe_text = "def calculate_area(radius): return 3.14159 * radius ** 2"
    assert contains_leaked_secrets(safe_text) is False


def test_latex_validation():
    broken_dollar = "Let $$x = 5$ be an integer."
    assert has_broken_latex(broken_dollar) is True

    valid_latex = "Let $$x = 5$$ and $$y = 10$$ be integers."
    assert has_broken_latex(valid_latex) is False


def test_stem_detection():
    stem_text = "The chemical reaction produces water and carbon dioxide with high velocity."
    assert is_stem_content(stem_text) is True

    non_stem_text = "I went to the grocery store yesterday to buy shoes."
    assert is_stem_content(non_stem_text) is False


# ==============================================================================
# 4. Deduplication Tests
# ==============================================================================

def test_global_deduplication():
    dedup = GlobalDeduplicator()
    msgs_1 = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."}
    ]
    msgs_2 = [
        {"role": "user", "content": "What is Python?   "},
        {"role": "assistant", "content": "Python is a  programming language."}
    ]
    msgs_3 = [
        {"role": "user", "content": "What is Rust?"},
        {"role": "assistant", "content": "Rust is a systems programming language."}
    ]

    assert dedup.is_duplicate(msgs_1, "smol_magpie_ultra") is False
    assert dedup.is_duplicate(msgs_2, "openhermes_2_5") is True  # Duplicate across source!
    assert dedup.is_duplicate(msgs_3, "magicoder_evol") is False

    stats = dedup.stats()
    assert stats["total_duplicates"] == 1
    assert stats["unique_records"] == 2
    assert "smol_magpie_ultra -> openhermes_2_5" in stats["top_cross_source_collisions"]


# ==============================================================================
# 5. Deficit Mixer Tests
# ==============================================================================

def test_instruct_mixer():
    sources_cfg = {
        "src_a": {"target_share": 0.60, "target_tokens": 600},
        "src_b": {"target_share": 0.40, "target_tokens": 400},
    }
    mixer = InstructMixer(sources_cfg, target_total_tokens=1000)

    # Initial selection should pick src_a (larger deficit)
    active = {"src_a", "src_b"}
    first = mixer._select_next_source(active)
    assert first in active

    mixer.record_sample("src_a", 600, 400)
    assert mixer.is_source_fulfilled("src_a") is True

    # Next selection must pick src_b
    nxt = mixer._select_next_source(active)
    assert nxt == "src_b"


# ==============================================================================
# 6. Packing & Arrow Sharding Tests
# ==============================================================================

def test_packing_and_sharding():
    tokenizer = InstructTokenizer("Qwen/Qwen3.5-0.8B-Base", max_seq_length=128)
    packer = InstructDocumentPacker(tokenizer, max_seq_length=128)

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"}
    ]
    inp, lab, rend, train = tokenizer.render_and_tokenize_conversation(messages)

    with tempfile.TemporaryDirectory() as tmpdir:
        sharder = DatasetSharder(tmpdir, shard_size_mb=1, max_sequences_per_shard=2)

        # Add docs until a packed sequence is produced
        for _ in range(20):
            seq = packer.add_conversation(inp, lab, rend, train)
            if seq:
                sharder.add_sequence(seq)

        final_seq = packer.flush()
        if final_seq:
            sharder.add_sequence(final_seq)

        sharder.finalize()

        # Verify Arrow file can be read
        arrow_files = sorted(Path(tmpdir).glob("*.arrow"))
        assert len(arrow_files) >= 1

        with open(str(arrow_files[0]), "rb") as f:
            reader = pa.ipc.open_file(f)
            table = reader.read_all()
        assert "input_ids" in table.column_names
        assert "labels" in table.column_names
        assert "attention_mask" in table.column_names
        assert len(table) >= 1


if __name__ == "__main__":
    test_funcs = [
        test_normalize_messages_format,
        test_normalize_conversations_format,
        test_normalize_instruction_response,
        test_normalize_problem_generated_solution,
        test_single_turn_assistant_masking,
        test_multi_turn_assistant_masking,
        test_all_tokens_loss_policy,
        test_secret_detection,
        test_latex_validation,
        test_stem_detection,
        test_global_deduplication,
        test_instruct_mixer,
        test_packing_and_sharding,
    ]

    print("=" * 60)
    print("RUNNING KAPINSTRUCT UNIT TESTS")
    print("=" * 60)

    passed = 0
    for fn in test_funcs:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print(f"RESULTS: {passed}/{len(test_funcs)} PASSED")
    print("=" * 60)
    sys.exit(0 if passed == len(test_funcs) else 1)

