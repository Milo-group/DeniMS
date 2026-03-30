# Model checkpoint path (required)
model_checkpoint_path = "/gpfs0/bgu-anatm/users/harniky/ms2mol/MS_diffusion/outputs/checkpoints/Fraghub_tempvar_random.ckpt"

# Experimental data path
experimental_parquet_path = "/gpfs0/bgu-anatm/users/harniky/DeniMS/Preprocessing/experimental/experimental.parquet"

# Encoder checkpoint path (optional if encoder weights are in main checkpoint)
encoder_checkpoint_path = None  # Set to None if encoder is in main checkpoint, or provide path

# Output directory for results
output_dir = "./inference_model_mixed"

# Number of inference repeats per compound (single-model mode)
num_repeats = 100

# Ensemble configuration (optional)
# If you provide a list of checkpoints here, ensemble sampling will be used
ensemble_model_checkpoints = ["/gpfs0/bgu-anatm/users/harniky/ms2mol/MS_diffusion/outputs/checkpoints/Fraghub_morgan_random.ckpt"]  # e.g. ["ckpt1.ckpt", "ckpt2.ckpt"]
repeats_per_model = 1000            # Number of repeats per model when using ensemble

# Batch size for inference
batch_size = 100

# Device (None = auto-detect, or specify 'cuda' or 'cpu')
device = 'cuda'

import sys
import os
import pathlib

current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

_ms2mol_root = pathlib.Path(current_dir).parents[1]
if str(_ms2mol_root) not in sys.path:
    sys.path.insert(0, str(_ms2mol_root))

from apply_model import run_inference_experimental

results = run_inference_experimental(
    model_checkpoint_path=model_checkpoint_path,
    experimental_parquet_path=experimental_parquet_path,
    encoder_checkpoint_path=encoder_checkpoint_path,
    output_dir=output_dir,
    num_repeats=num_repeats,
    batch_size=batch_size,
    device=device,
    ensemble_model_checkpoints=ensemble_model_checkpoints,
    repeats_per_model=repeats_per_model,
)

print("\nInference completed successfully!")