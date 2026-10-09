"""Matched MLX/PyTorch inference and local HTTP serving comparison."""
import os

# MLX defaults can use reduced-precision matrix units with FP32 arrays.
# This experiment explicitly compares full FP32 on both backends.
os.environ['MLX_ENABLE_TF32'] = '0'
