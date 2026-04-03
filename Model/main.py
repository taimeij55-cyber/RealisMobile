import os
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL

# ============================================================================
# main.py: 从checkpoint恢复Realistic Vision V5.1模型并导出ONNX (FP16)
# ============================================================================

def _safe_onnx_export(*args, **kwargs):
    try:
        torch.onnx.export(*args, **kwargs)
    except TypeError as e:
        if "dynamo" in str(e):
            kwargs.pop("dynamo", None)
            torch.onnx.export(*args, **kwargs)
        else:
            raise

def restore_and_export(model_name="SG161222/Realistic_Vision_V5.1_noVAE",
                       output_dir="./ONNX"):
    os.makedirs(output_dir, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        model_name, torch_dtype=torch.float16,
        safety_checker=None, requires_safety_checker=False
    )

    # 替换VAE为sd-vae-ft-mse (Realistic Vision作者推荐)
    try:
        better_vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse", torch_dtype=torch.float16
        )
        pipe.vae = better_vae
    except Exception as e:
        print(f"VAE replacement failed: {e}")

    pipe = pipe.to("cpu")
    model_dtype = torch.float16

    te_hidden_dim = pipe.text_encoder.config.hidden_size
    print(f"Text Encoder hidden dim: {te_hidden_dim}")

    # 导出Text Encoder
    te_dir = os.path.join(output_dir, "text_encoder")
    os.makedirs(te_dir, exist_ok=True)
    dummy_ids = torch.randint(0, 49408, (1, 77), dtype=torch.long)
    _safe_onnx_export(
        pipe.text_encoder, (dummy_ids,),
        os.path.join(te_dir, "model.onnx"),
        input_names=["input_ids"],
        output_names=["last_hidden_state", "pooler_output"],
        dynamic_axes={"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False
    )

    # 导出UNet
    unet_dir = os.path.join(output_dir, "unet")
    os.makedirs(unet_dir, exist_ok=True)
    dummy_latent = torch.randn(1, 4, 64, 64, dtype=model_dtype)
    dummy_ts = torch.tensor([999], dtype=torch.long)
    dummy_enc = torch.randn(1, 77, te_hidden_dim, dtype=model_dtype)
    _safe_onnx_export(
        pipe.unet, (dummy_latent, dummy_ts, dummy_enc),
        os.path.join(unet_dir, "model.onnx"),
        input_names=["sample", "timestep", "encoder_hidden_states"],
        output_names=["out_sample"],
        dynamic_axes={"sample": {0: "batch"}, "encoder_hidden_states": {0: "batch"}, "out_sample": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False
    )

    # 导出VAE Decoder
    vae_dir = os.path.join(output_dir, "vae_decoder")
    os.makedirs(vae_dir, exist_ok=True)
    class VAEDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae
        def forward(self, latent):
            return self.vae.decode(latent).sample
    vae_dec = VAEDecoder(pipe.vae)
    dummy_vae = torch.randn(1, 4, 64, 64, dtype=model_dtype)
    _safe_onnx_export(
        vae_dec, (dummy_vae,),
        os.path.join(vae_dir, "model.onnx"),
        input_names=["latent"], output_names=["image"],
        dynamic_axes={"latent": {0: "batch"}, "image": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False
    )

    pipe.tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))
    print(f"ONNX models exported to {output_dir}")

if __name__ == "__main__":
    restore_and_export()
