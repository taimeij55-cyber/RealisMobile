RealisMobile: Efficient Stable Diffusion on Apple Silicon
MAI 2026 Challenge – Track 2 (Unconstrained)

RealisMobile is an optimized deployment pipeline for Stable Diffusion on Apple Silicon (M4), focusing on efficient on-device inference with Core ML acceleration. By integrating ONNX conversion and split-einsum attention reformulation, the system achieves full hardware utilization (GPU + ANE) and significant speed improvements over standard PyTorch pipelines.

🚀 Key Features
⚡ Core ML Acceleration: Up to 1.53× faster than PyTorch MPS (15s vs 23s per image @512×512)
🔄 Multi-backend Support: PyTorch / ONNX / Core ML
🧠 Split-Einsum Attention: Eliminates CPU fallback and ensures full operator compatibility
🎯 Golden Seed Strategy: Generate–evaluate–select framework for higher-quality outputs
📱 Edge Deployment Ready: Optimized for Apple M4 (GPU + Neural Engine)
📦 Requirements
pip install diffusers transformers torch onnxruntime coremltools numpy Pillow
🧠 Method Overview
Base Configuration
Model: Realistic Vision V5.1
VAE: sd-vae-ft-mse
Scheduler: DPM++ 2M Karras
Guidance Scale: 7.0
Inference Steps:
30 (development / Kaggle T4)
50 (deployment / Apple M4)
Optimization Pipeline
PyTorch → ONNX (FP16) → Core ML
Split-einsum attention (CoreML-compatible)
Multi-backend execution (CPU / GPU / ANE)
Golden Seed Strategy
Generate 8 candidates per prompt
Select the best result using quality heuristics
⚙️ Performance Benchmark (Apple M4)
Backend	Device	Time / Image
PyTorch FP32	CPU	175 s
PyTorch FP16 (MPS)	GPU	23 s
ONNX FP16	CPU	68 s
Core ML (FP16)	GPU + ANE	15 s
📊 Design Choices
❌ No FreeU / SAG / prompt tricks
✅ Focus on system-level optimization
❌ INT8 quantization (causes >15% quality drop)
✅ FP16 for stable generation quality
📁 Repository Contents
convert/ – PyTorch → ONNX → Core ML pipeline
inference/ – Optimized inference scripts
benchmark/ – Multi-backend evaluation tools
results/ – Sample generated images
🎯 Goal

Enable practical and high-quality diffusion model deployment on edge devices, providing both a reproducible engineering pipeline and insights into real-world optimization of large-scale generative models.
