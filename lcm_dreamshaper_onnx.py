import os
import time
from pathlib import Path
from huggingface_hub import snapshot_download, login
from optimum.onnxruntime import ORTStableDiffusionPipeline
from diffusers.schedulers import LCMScheduler
from PIL import Image
from output_utils import ensure_output_dirs, save_image, log_timing
from prompt_config import DEFAULT_PROMPT

# =========================
# Configurações
# =========================
# Repositório que contém os componentes ONNX do LCM Dreamshaper v7.
# Este repo comunitário expõe UNet/TextEncoder/VAE já convertidos para ONNX.
HF_REPO = "SimianLuo/LCM_Dreamshaper_v7"  # ajuste se desejar outro
LOCAL_DIR = Path("./models/LCM_Dreamshaper_v7-onnx")  # pasta local
# Provider do ONNX Runtime: "CUDAExecutionProvider", "CPUExecutionProvider",
# "ROCMExecutionProvider", "DmlExecutionProvider" (Windows)
# Padrão alterado para CPUExecutionProvider; sobrescreva com a variável ORT_PROVIDER se quiser GPU.
ORT_PROVIDER = os.getenv("ORT_PROVIDER", "CPUExecutionProvider")

# Prompt: se LCM_PROMPT não estiver definido, usa o DEFAULT_PROMPT de prompt_config.py
PROMPT = os.getenv("LCM_PROMPT") or DEFAULT_PROMPT
NEGATIVE_PROMPT = "low quality, blurry, deformed"
NUM_STEPS = int(os.getenv("LCM_STEPS", "50"))           # LCM funciona bem com 1–8
GUIDANCE = float(os.getenv("LCM_GUIDANCE", "8.0"))     # sugestão comum
# Resolução da imagem (reduza se estiver com pouco RAM/VRAM)
HEIGHT = int(os.getenv("LCM_HEIGHT", "512"))
WIDTH = int(os.getenv("LCM_WIDTH", "512"))

# =========================
# Download se não existir
# =========================
def ensure_model(repo_id: str, local_dir: Path):
    """
    Baixa o snapshot do repo para local_dir se ainda não existir.
    Requer git-lfs no ambiente do HF Hub (feito automaticamente pelo snapshot_download).
    """
    local_dir.mkdir(parents=True, exist_ok=True)

    # Verifica presença dos principais componentes ONNX
    needed = [
        local_dir / "unet" / "model.onnx",
        local_dir / "text_encoder" / "model.onnx",
        local_dir / "vae_decoder" / "model.onnx",
    ]
    if all(p.exists() for p in needed):
        print(f"[OK] Modelo já presente em: {local_dir.resolve()}")
        return str(local_dir)

    print(f"[INFO] Baixando modelo ONNX de {repo_id} para {local_dir} ...")
    # Se você precisar de token (modelo privado), descomente:
    # login(token=os.getenv("HF_TOKEN"))
    snapshot_download(repo_id=repo_id, local_dir=str(local_dir), local_dir_use_symlinks=False)
    print("[OK] Download concluído.")
    return str(local_dir)

# =========================
# Carregar pipeline ORT + LCM
# =========================
def load_pipeline_onnx(model_dir: str, provider: str):
    print(f"[INFO] Carregando ORTStableDiffusionPipeline com provider={provider} ...")
    pipe = ORTStableDiffusionPipeline.from_pretrained(
        model_dir,
        provider=provider,
    )
    # Substitui o scheduler padrão pelo LCM (essencial para few-step)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    return pipe

def main():
    model_path = ensure_model(HF_REPO, LOCAL_DIR)
    pipe = load_pipeline_onnx(model_path, ORT_PROVIDER)

    # Geração
    print(f"[INFO] Gerando imagem com {NUM_STEPS} passos e guidance={GUIDANCE} ...")

    ensure_output_dirs()

    start = time.time()
    image = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE,
        height=HEIGHT,
        width=WIDTH,
    ).images[0]
    end = time.time()
    duration = end - start

    image_path, timestamp = save_image(image, prefix="lcm_dreamshaper")
    log_timing(
        timestamp=timestamp,
        duration=duration,
        image_path=image_path,
        num_steps=NUM_STEPS,
        log_filename="lcm_timings.csv",
    )

    print(f"[OK] Imagem salva em: {image_path}")
    print(f"[INFO] Tempo de geração: {duration:.4f} s")

if __name__ == "__main__":
    main()
