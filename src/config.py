"""
Model configuration constants for DeepSeek-OCR-2.
No external dependencies — safe to import anywhere.
"""

# Image preprocessing
BASE_SIZE = 1024       # Global view size (padded to this)
IMAGE_SIZE = 768       # Local crop tile size
CROP_MODE = True
MIN_CROPS = 2
MAX_CROPS = 6

# Default model path (HuggingFace Hub)
MODEL_PATH = "deepseek-ai/DeepSeek-OCR-2"

# Default prompt
PROMPT = "<image>\n<|grounding|>Convert the document to markdown."

# Vision encoder defaults
SAM_EMBED_DIM = 768
SAM_DEPTH = 12
SAM_NUM_HEADS = 12
SAM_GLOBAL_ATTN_INDEXES = [2, 5, 8, 11]
SAM_WINDOW_SIZE = 14
SAM_OUT_CHANS = 256   # after neck

# D2E defaults
D2E_LAYERS = 24
D2E_HIDDEN_DIM = 896
D2E_NUM_HEADS = 14
D2E_NUM_KV_HEADS = 2
D2E_INTERMEDIATE_SIZE = 4864

# Projector
PROJECTOR_INPUT_DIM = 896
PROJECTOR_N_EMBED = 1280

# Special tokens
IMAGE_TOKEN = "<image>"
