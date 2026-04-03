# Realistic Vision V5.1 Efficient Stable Diffusion V7

## MAI 2026 Challenge - Track 2 (Unconstrained)

## Requirements

# RealisMobile
RealisMobile is an optimized pipeline for deploying Stable Diffusion on Apple Silicon (M4), integrating PyTorch, ONNX, and Core ML for efficient on-device generation. With split-einsum attention and multi-backend acceleration, it achieves up to 1.53× speedup over PyTorch MPS.


```
pip install diffusers transformers torch onnxruntime numpy Pillow
```

## Method
- Base model: Realistic Vision V5.1 (SG161222/Realistic_Vision_V5.1_noVAE)
- Scheduler: DPM++ 2M Karras (author recommended)
- Inference steps: 30 (Kaggle), 50 (ONNX)
- Guidance scale: 7.0 (optimized via multi-model comparison testing)
- Golden Seed: 8 candidates per prompt
- VAE: sd-vae-ft-mse (author recommended)
- No FreeU, no SAG, no prompt prefix (back to basics)
- ONNX precision: FP16
- Target: Apple M4 Neural Engine (CPU)
