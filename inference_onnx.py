import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from transformers import CLIPTokenizer

# ============================================================================
# Realistic Vision V5.1 ONNX推理脚本 (V7: CFG引导 + DDIM 50步)
# 加载三个ONNX组件: Text Encoder + UNet + VAE Decoder
# 自动适配FP16/FP32 ONNX模型输入类型
# ============================================================================

class SD15ONNXPipeline:
    def __init__(self, onnx_dir):
        """初始化ONNX推理管线"""
        tokenizer_dir = os.path.join(onnx_dir, "tokenizer")
        if os.path.exists(tokenizer_dir):
            self.tokenizer = CLIPTokenizer.from_pretrained(tokenizer_dir)
        else:
            self.tokenizer = CLIPTokenizer.from_pretrained(
                "SG161222/Realistic_Vision_V5.1_noVAE", subfolder="tokenizer"
            )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = ["CPUExecutionProvider"]
        available = ort.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        te_path = os.path.join(onnx_dir, "text_encoder", "model.onnx")
        unet_path = os.path.join(onnx_dir, "unet", "model.onnx")
        vae_path = os.path.join(onnx_dir, "vae_decoder", "model.onnx")

        self.text_encoder = ort.InferenceSession(te_path, opts, providers=providers)
        self.unet = ort.InferenceSession(unet_path, opts, providers=providers)
        self.vae_decoder = ort.InferenceSession(vae_path, opts, providers=providers)

        self.float_dtype = self._detect_float_dtype(self.unet)
        print(f"ONNX float type: {self.float_dtype}")

        # V7参数: 50步DDIM + CFG=7.0 (多模型对比测试最优)
        self.num_steps = 50
        self.guidance_scale = 7.0

        self._init_scheduler()

    def _detect_float_dtype(self, session):
        for inp in session.get_inputs():
            if inp.type == "tensor(float16)":
                return np.float16
            elif inp.type == "tensor(float)":
                return np.float32
        return np.float32

    def _init_scheduler(self):
        beta_start, beta_end = 0.00085, 0.012
        num_train_timesteps = 1000
        betas = np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps)**2
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas).astype(np.float64)

    def _get_timesteps(self, num_steps):
        step_ratio = 1000 // num_steps
        timesteps = np.arange(999, -1, -step_ratio, dtype=np.int64)[:num_steps]
        return timesteps

    def encode_prompt(self, prompt):
        tokens = self.tokenizer(
            prompt, padding="max_length", max_length=77,
            truncation=True, return_tensors="np"
        )
        input_ids = tokens["input_ids"].astype(np.int64)
        result = self.text_encoder.run(None, {"input_ids": input_ids})
        return result[0]

    def denoise(self, latent, timestep, encoder_hidden_states):
        result = self.unet.run(None, {
            "sample": latent.astype(self.float_dtype),
            "timestep": np.array([timestep], dtype=np.int64),
            "encoder_hidden_states": encoder_hidden_states.astype(self.float_dtype)
        })
        return result[0]

    def decode_latent(self, latent):
        latent = latent / 0.18215
        vae_dtype = self._detect_float_dtype(self.vae_decoder)
        result = self.vae_decoder.run(None, {"latent": latent.astype(vae_dtype)})
        return result[0]

    def ddim_step(self, noise_pred, timestep, sample, next_timestep=None):
        noise_pred_f64 = noise_pred.astype(np.float64)
        sample_f64 = sample.astype(np.float64)

        alpha_t = self.alphas_cumprod[timestep]
        sqrt_alpha_t = np.sqrt(alpha_t)
        sqrt_one_minus_alpha_t = np.sqrt(1.0 - alpha_t)

        pred_x0 = (sample_f64 - sqrt_one_minus_alpha_t * noise_pred_f64) / sqrt_alpha_t

        if next_timestep is None:
            return pred_x0.astype(np.float32)

        alpha_next = self.alphas_cumprod[next_timestep]
        prev_sample = (np.sqrt(alpha_next) * pred_x0 +
                       np.sqrt(1.0 - alpha_next) * noise_pred_f64)
        return prev_sample.astype(np.float32)

    def generate(self, prompt, negative_prompt="", seed=42, num_steps=None,
                 guidance_scale=None):
        if num_steps is None:
            num_steps = self.num_steps
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        np.random.seed(seed)

        # V7: 直接使用原始prompt, 不加任何前缀
        cond_emb = self.encode_prompt(prompt)
        if guidance_scale > 1.0:
            uncond_emb = self.encode_prompt(negative_prompt if negative_prompt else "")
        else:
            uncond_emb = None

        latent = np.random.randn(1, 4, 64, 64).astype(np.float32)
        timesteps = self._get_timesteps(num_steps)

        for i, t in enumerate(timesteps):
            if uncond_emb is not None and guidance_scale > 1.0:
                noise_pred_cond = self.denoise(latent, int(t), cond_emb)
                noise_pred_uncond = self.denoise(latent, int(t), uncond_emb)
                noise_pred = (noise_pred_uncond.astype(np.float64) +
                              guidance_scale * (noise_pred_cond.astype(np.float64) -
                                                noise_pred_uncond.astype(np.float64)))
                noise_pred = noise_pred.astype(np.float32)
            else:
                noise_pred = self.denoise(latent, int(t), cond_emb)

            next_t = int(timesteps[i + 1]) if i + 1 < len(timesteps) else None
            latent = self.ddim_step(noise_pred, int(t), latent, next_t)

        image = self.decode_latent(latent)
        image = np.clip((image + 1.0) / 2.0 * 255.0, 0, 255).astype(np.uint8)
        image = image[0].transpose(1, 2, 0)
        return Image.fromarray(image)


# Realistic Vision V5.1 作者推荐Negative Prompt
DEFAULT_NEGATIVE = (
    '(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, '
    'cartoon, drawing, anime:1.4), text, close up, cropped, out of frame, '
    'worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, '
    'mutilated, extra fingers, mutated hands, poorly drawn hands, '
    'poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, '
    'bad proportions, extra limbs, cloned face, disfigured, gross proportions, '
    'malformed limbs, missing arms, missing legs, extra arms, extra legs, '
    'fused fingers, too many fingers, long neck'
)


if __name__ == "__main__":
    import sys
    import time
    onnx_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    prompt = sys.argv[2] if len(sys.argv) > 2 else "A beautiful sunset"
    output = sys.argv[3] if len(sys.argv) > 3 else "output.png"
    pipeline = SD15ONNXPipeline(onnx_dir)
    start = time.time()
    image = pipeline.generate(prompt, negative_prompt=DEFAULT_NEGATIVE, num_steps=50)
    elapsed = time.time() - start
    image.save(output)
    print(f"Generated {output} in {elapsed:.2f}s")
