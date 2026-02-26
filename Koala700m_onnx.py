from diffusers import StableDiffusionXLPipeline
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np
from PIL import Image
import torch
import time
import os
from report_utils import (
    LOGS_DIR,
    IMAGES_DIR,
    ensure_directories,
    get_next_test_number,
    get_machine_specs,
    build_base_report_lines,
    finalize_and_write_report,
)


MODEL_ID = "etri-vilab/koala-700m"
MODEL_NAME = MODEL_ID.split("/")[-1]
VAE_SCALING_FACTOR = 0.13025  # from vae_decoder/config.json


def decode_latents_with_onnx_vae(latents: torch.Tensor) -> Image.Image:
    """Decode latents using the KOALA VAE decoder exported to ONNX."""

    if latents.ndim == 3:
        latents = latents.unsqueeze(0)

    latents = latents / VAE_SCALING_FACTOR
    latents_np = latents.detach().cpu().numpy().astype("float32")

    onnx_path = hf_hub_download(repo_id=MODEL_ID, filename="vae_decoder/model.onnx")

    provider = "CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"
    sess = ort.InferenceSession(onnx_path, providers=[provider])

    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: latents_np})

    image_np = outputs[0]
    image_np = (image_np / 2.0 + 0.5).clip(0.0, 1.0)

    image_np = np.clip(image_np * 255.0, 0, 255).astype("uint8")
    if image_np.ndim == 4:
        image_np = image_np[0]
    image_np = np.transpose(image_np, (1, 2, 0))

    return Image.fromarray(image_np)


def main():
    ensure_directories()

    variant_name = f"{MODEL_NAME}_onnxvae"
    test_number = get_next_test_number(variant_name)
    test_tag = f"{test_number:04d}_{variant_name}"

    image_path = os.path.join(IMAGES_DIR, f"{test_tag}.png")
    log_path = os.path.join(LOGS_DIR, f"{test_tag}.log")

    start_total = time.perf_counter()

    # PyTorch pipeline for UNet/text encoders
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionXLPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe = pipe.to(device)
    t1 = time.perf_counter()

    prompt = (
        "Cute and colorful birthday card featuring an adorable black cat. "
        "Vibrant digital illustration style, clean linework, centered composition, "
        "balloons and confetti in the background, soft lighting, cheerful colors, "
        "light background, greeting card aesthetic."
    )

    num_inference_steps = 25

    # Generate latents only
    t2 = time.perf_counter()
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=7.5,
        output_type="latent",
    )
    latents = result.images[0]

    # Decode latents with ONNX VAE
    image = decode_latents_with_onnx_vae(latents)
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    image.save(image_path)
    t5 = time.perf_counter()

    end_total = time.perf_counter()

    timings = {
        "load_model_s": t1 - t0,
        "inference_latency_s": t3 - t2,
        "save_image_s": t5 - t4,
        "total_time_s": end_total - start_total,
    }

    specs = get_machine_specs()

    lines = build_base_report_lines(
        test_number=test_number,
        model_id=f"{MODEL_ID} (PyTorch UNet + ONNX VAE)",
        test_tag=test_tag,
        image_path=image_path,
        log_path=log_path,
        specs=specs,
    )
    lines.append("===== TIMINGS (SECONDS) =====")
    lines.append(f"Load model: {timings['load_model_s']:.3f}s")
    lines.append(
        f"Inference latency (PyTorch + ONNX VAE): {timings['inference_latency_s']:.3f}s"
    )
    lines.append(f"Save image: {timings['save_image_s']:.3f}s")
    lines.append(f"Total time: {timings['total_time_s']:.3f}s")

    finalize_and_write_report(log_path, lines)


if __name__ == "__main__":
    main()
