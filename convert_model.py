#!/usr/bin/env python3
"""
Convert GTE-small model from safetensors to .gtemodel binary format (4-bit Quantized).

Usage:
    python convert_model.py offline/local_complete_model gte-small.gtemodel
"""

import sys
import struct
import json
import numpy as np
from pathlib import Path

try:
    from safetensors import safe_open
except ImportError:
    print("Please install safetensors: pip install safetensors")
    sys.exit(1)

def quantize_q4_0(tensor, block_size=32):
    orig_shape = tensor.shape
    flattened = tensor.flatten()

    n = flattened.size
    padding = (block_size - (n % block_size)) % block_size
    if padding > 0:
        flattened = np.concatenate([flattened, np.zeros(padding, dtype='float32')])

    reshaped = flattened.reshape(-1, block_size)

    scales = np.max(np.abs(reshaped), axis=1) / 7.0
    scales = scales.astype('float32')

    inv_scales = np.where(scales != 0, 1.0 / scales, 0).reshape(-1, 1)
    quant = np.round(reshaped * inv_scales).clip(-8, 7).astype(np.int8)

    # FIX: Shift values by +8 so they sit in the 0..15 range for the C code
    quant = (quant + 8).astype(np.uint8)

    packed = np.zeros((quant.shape[0], block_size // 2), dtype=np.uint8)
    for i in range(block_size // 2):
        # FIX: Simply OR them together now
        packed[:, i] = quant[:, i*2] | (quant[:, i*2+1] << 4)

    return scales, packed

def load_vocab(vocab_path):
    """Load vocabulary from vocab.txt"""
    vocab = []
    with open(vocab_path, 'r', encoding='utf-8') as f:
        for line in f:
            vocab.append(line.rstrip('\n'))
    return vocab

def main():
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <model_dir> <output.gtemodel>")
        sys.exit(1)

    model_dir = Path(sys.argv[1])
    output_path = sys.argv[2]

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    vocab_size = config["vocab_size"]
    hidden_size = config["hidden_size"]
    num_layers = config["num_hidden_layers"]
    num_heads = config["num_attention_heads"]
    intermediate_size = config["intermediate_size"]
    max_seq_length = config["max_position_embeddings"]

    print(f"Model config (Quantized):")
    print(f"  vocab_size: {vocab_size} | hidden_size: {hidden_size}")
    print(f"  num_layers: {num_layers} | num_heads: {num_heads}")

    vocab = load_vocab(model_dir / "vocab.txt")
    safetensors_path = model_dir / "model.safetensors"
    tensors = safe_open(safetensors_path, framework="numpy")

    with open(output_path, 'wb') as f:
        # Header (Updated Magic)
        f.write(b'GTE4')
        f.write(struct.pack('<I', vocab_size))
        f.write(struct.pack('<I', hidden_size))
        f.write(struct.pack('<I', num_layers))
        f.write(struct.pack('<I', num_heads))
        f.write(struct.pack('<I', intermediate_size))
        f.write(struct.pack('<I', max_seq_length))

        # Vocabulary
        for word in vocab:
            word_bytes = word.encode('utf-8')
            f.write(struct.pack('<H', len(word_bytes)))
            f.write(word_bytes)

        def write_tensor(name):
            tensor = tensors.get_tensor(name).astype('float32')
            f.write(tensor.tobytes())
            return tensor.shape

        def write_tensor_q4(name):
            tensor = tensors.get_tensor(name).astype('float32')
            # Only quantize 2D weight matrices (Linear layers)
            if len(tensor.shape) == 2:
                scales, packed = quantize_q4_0(tensor)
                # Write blocks: interleaved scale (4B) and packed bytes (16B)
                for s, p in zip(scales, packed):
                    f.write(s.tobytes())
                    f.write(p.tobytes())
            else:
                # Fallback for biases/LayerNorms/Embeddings if passed here
                f.write(tensor.tobytes())
            return tensor.shape

        # Embeddings (Quantized with Q4_0)
        print("\nWriting embeddings (4-bit)...")
        write_tensor_q4("embeddings.word_embeddings.weight")
        write_tensor_q4("embeddings.position_embeddings.weight")
        write_tensor_q4("embeddings.token_type_embeddings.weight")
        write_tensor("embeddings.LayerNorm.weight")
        write_tensor("embeddings.LayerNorm.bias")

        # Transformer layers
        print("Writing transformer layers (4-bit)...")
        for layer_idx in range(num_layers):
            prefix = f"encoder.layer.{layer_idx}"

            # Weights: Quantized | Biases: FP32
            write_tensor_q4(f"{prefix}.attention.self.query.weight")
            write_tensor(f"{prefix}.attention.self.query.bias")
            write_tensor_q4(f"{prefix}.attention.self.key.weight")
            write_tensor(f"{prefix}.attention.self.key.bias")
            write_tensor_q4(f"{prefix}.attention.self.value.weight")
            write_tensor(f"{prefix}.attention.self.value.bias")
            write_tensor_q4(f"{prefix}.attention.output.dense.weight")
            write_tensor(f"{prefix}.attention.output.dense.bias")

            write_tensor(f"{prefix}.attention.output.LayerNorm.weight")
            write_tensor(f"{prefix}.attention.output.LayerNorm.bias")

            write_tensor_q4(f"{prefix}.intermediate.dense.weight")
            write_tensor(f"{prefix}.intermediate.dense.bias")
            write_tensor_q4(f"{prefix}.output.dense.weight")
            write_tensor(f"{prefix}.output.dense.bias")

            write_tensor(f"{prefix}.output.LayerNorm.weight")
            write_tensor(f"{prefix}.output.LayerNorm.bias")

            print(f"  Layer {layer_idx} quantized and saved")

        # Pooler
        write_tensor_q4("pooler.dense.weight")
        write_tensor("pooler.dense.bias")

    print(f"\nModel saved to {output_path}")
    print(f"Final size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main()
