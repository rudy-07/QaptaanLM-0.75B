# KapInstruct-100M Dataset Card

**KapInstruct-100M** is a high-fidelity, multi-source instruction tuning dataset containing **0 usable content tokens** formatted with Qwen ChatML and tokenized using `Qwen/Qwen3.5-0.8B-Base`.

## Dataset Summary
- **Total Rendered Tokens**: 51,751
- **Trainable Assistant Tokens**: 43,677
- **Total Documents / Conversations**: 51
- **Sequence Length**: 4096
- **Number of Shards**: 1
- **Loss Masking Policy**: `assistant_only` (Loss computed strictly on assistant turns; prompt tokens masked to `-100`)

## Mixture Composition & Source-Specific Licenses

| Source | Domain | Share Target | Rendered Tokens | Trainable Tokens | License | Pinned Commit SHA |
|--------|--------|--------------|-----------------|------------------|---------|-------------------|
| `code_debugging` | code_debugging | 10% | 1,448 | 1,057 | Apache-2.0 | `a08c213a97...` |
| `magicoder_evol` | magicoder_evol | 13% | 2,291 | 1,697 | Apache-2.0 | `b0079beaa0...` |
| `magicoder_oss` | magicoder_oss | 8% | 2,226 | 1,131 | MIT | `5f839b1f36...` |
| `numinamath_cot` | numinamath_cot | 6% | 2,154 | 1,737 | Apache-2.0 | `9d8d210c9f...` |
| `openhermes_2_5` | openhermes_2_5 | 9% | 1,506 | 1,247 | MIT / Open | `b820378210...` |
| `openmathinstruct2` | openmathinstruct2 | 11% | 2,694 | 2,041 | CC-BY-4.0 | `469216e3f4...` |
| `openthoughts_reasoning` | openthoughts_reasoning | 7% | 31,530 | 28,649 | Apache-2.0 | `bd093c3994...` |
| `self_oss_starcoder2` | self_oss_starcoder2 | 5% | 463 | 329 | ODC-By | `356bb069ee...` |
| `smol_constraints` | smol_constraints | 3% | 913 | 423 | Apache-2.0 / Open | `5feaf2fd3f...` |
| `smol_magpie_ultra` | smol_magpie_ultra | 18% | 5,579 | 4,515 | Apache-2.0 / Open | `5feaf2fd3f...` |
| `stem_qa` | stem_qa | 4% | 947 | 851 | Apache-2.0 | `559b33b6bc...` |
| `tulu3_sft` | tulu3_sft | 6% | 0 | 0 | ODC-By | `b14afda60f...` |

## Licensing and Provenance Notice
Each individual subset in this mixture retains its own upstream license as listed above. Users and researchers must adhere to the individual terms of each constituent source (e.g. CC-BY-4.0 attribution for OpenMathInstruct-2, ODC-By for Tulu-3 / Self-OSS, Apache-2.0, MIT). No single overarching permissive license is claimed over the composite corpus.

## Loading & Inspection Example
```python
import pyarrow as pa
import glob

# Load Arrow shards directly
shard_files = sorted(glob.glob("data/kapinstruct/*.arrow"))
for shard_path in shard_files[:1]:
    reader = pa.ipc.open_file(shard_path)
    table = reader.read_all()
    print(f"Loaded {len(table)} sequences from {shard_path}")
    print("Columns:", table.column_names)
```
