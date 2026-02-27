import os
import time
from pathlib import Path

import torch
from diffusers import DiffusionPipeline, LCMScheduler
from output_utils import ensure_output_dirs, save_image, log_timing
from prompt_config import DEFAULT_PROMPT

# =========================
# Configurações
# =========================
HF_REPO = "SimianLuo/LCM_Dreamshaper_v7"
LOCAL_DIR = Path("./models/LCM_Dreamshaper_v7-torch")

# Dispositivo: "cuda" se disponível, caso contrário "cpu".
DEVICE = os.getenv("LCM_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if os.getenv("LCM_FP16", "0") == "1" and DEVICE == "cuda" else torch.float32

PROMPT = os.getenv("LCM_PROMPT") or DEFAULT_PROMPT
NEGATIVE_PROMPT = os.getenv("LCM_NEGATIVE", "low quality, blurry, deformed")
NUM_STEPS = int(os.getenv("LCM_STEPS", "8"))
GUIDANCE = float(os.getenv("LCM_GUIDANCE", "8.0"))
HEIGHT = int(os.getenv("LCM_HEIGHT", "512"))
WIDTH = int(os.getenv("LCM_WIDTH", "512"))


def ensure_model(repo_id: str, local_dir: Path) -> str:
    """Garante que o modelo PyTorch esteja baixado localmente."""
    local_dir.mkdir(parents=True, exist_ok=True)
    # Diffusers baixa os arquivos automaticamente na primeira chamada de from_pretrained,
    # mas usamos um diretório dedicado para manter organizado.
    return str(local_dir)


def load_pipeline_torch(model_dir: str):
    """Carrega o pipeline Diffusers padrão (não ONNX) com LCM scheduler."""
    print(f"[INFO] Carregando DiffusionPipeline em {DEVICE} (dtype={DTYPE}) ...")
    pipe = DiffusionPipeline.from_pretrained(
        HF_REPO,
        torch_dtype=DTYPE,
        cache_dir=model_dir,
        safety_checker=None,  # opcional: desative se quiser ligeiro ganho de performance
    )

    # Substitui o scheduler pelo LCM, como no exemplo oficial
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    pipe.to(DEVICE)
    return pipe


def main():
    model_path = ensure_model(HF_REPO, LOCAL_DIR)
    pipe = load_pipeline_torch(model_path)

    print(f"[INFO] Gerando imagem Torch LCM com {NUM_STEPS} passos, guidance={GUIDANCE}, {WIDTH}x{HEIGHT} ...")

    ensure_output_dirs()

    generator = torch.Generator(device=DEVICE)

    start = time.time()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    ).images[0]
    end = time.time()
    duration = end - start

    image_path, timestamp = save_image(image, prefix="lcm_dreamshaper_torch")
    log_timing(
        timestamp=timestamp,
        duration=duration,
        image_path=image_path,
        num_steps=NUM_STEPS,
        log_filename="lcm_torch_timings.csv",
    )

    print(f"[OK] Imagem salva em: {image_path}")
    print(f"[INFO] Tempo de geração: {duration:.4f} s")


if __name__ == "__main__":
    main()
