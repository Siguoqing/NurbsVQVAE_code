
import torch
import sys
import os

# Put current path in sys.path
sys.path.append(os.getcwd())

from model import LLaMA3ARModel, LLaMA3Config

def test_vectorization():
    print("Testing Vectorized Log Probs...")
    
    # 1. Init Config
    config = LLaMA3Config(
        vocab_size=1000,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        dim_feedforward=512,
        pad_token_id=0
    )
    
    # 2. Init Model
    model = LLaMA3ARModel(config)
    
    # 3. Dummy Data
    batch_size = 2
    seq_len = 10
    # Random input ids from [1, vocab_size]
    input_ids = torch.randint(1, 1000, (batch_size, seq_len, 3))
    
    # Ensure no PAD tokens for this test
    
    # 4. Run Vectorized
    try:
        log_probs = model.compute_all_log_probs(input_ids)
        print(f"Success! Output shape: {log_probs.shape}")
        
        expected_shape = (batch_size, seq_len - 1)
        if log_probs.shape == expected_shape:
             print("Shape check passed.")
        else:
             print(f"Shape check FAILED. Expected {expected_shape}, got {log_probs.shape}")
             
    except Exception as e:
        print(f"Vectorized computation FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_vectorization()
