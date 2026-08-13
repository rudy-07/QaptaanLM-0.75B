"""Qwen3.5-0.8B CPT+SFT Fine-Tuning Pipeline."""

# Patch torchvision check in transformers to prevent broken C-extension crashes in Colab/Kaggle
try:
    import transformers.utils.import_utils as _iu
    _iu._torchvision_available = False
    _iu.is_torchvision_available = lambda: False
except Exception:
    pass

