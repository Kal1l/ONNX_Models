import os
import time
import platform

try:
    import psutil  # type: ignore
except ImportError:  # psutil é opcional
    psutil = None


IMAGES_DIR = "images"
LOGS_DIR = "logs"


def ensure_output_dirs(images_dir: str = IMAGES_DIR, logs_dir: str = LOGS_DIR) -> None:
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


def save_image(image, images_dir: str = IMAGES_DIR, prefix: str = "sdxl_turbo"):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{prefix}_{timestamp}.png"
    image_path = os.path.join(images_dir, filename)
    image.save(image_path)
    return image_path, timestamp


def _get_system_info() -> tuple[str, str, float | None, float | None]:
    """Retorna (os, cpu, total_ram_mb, used_ram_mb) se possível."""
    os_name = f"{platform.system()} {platform.release()}"
    cpu_name = platform.processor() or platform.machine()

    total_ram_mb: float | None = None
    used_ram_mb: float | None = None

    if psutil is not None:
        try:
            vm = psutil.virtual_memory()
            total_ram_mb = vm.total / (1024 * 1024)
            proc = psutil.Process(os.getpid())
            used_ram_mb = proc.memory_info().rss / (1024 * 1024)
        except Exception:
            pass

    return os_name, cpu_name, total_ram_mb, used_ram_mb


def log_timing(
    timestamp: str,
    duration: float,
    image_path: str,
    num_steps: int | None = None,
    logs_dir: str = LOGS_DIR,
    log_filename: str = "timings.csv",
) -> None:
    os_name, cpu_name, total_ram_mb, used_ram_mb = _get_system_info()

    log_path = os.path.join(logs_dir, log_filename)
    is_new = not os.path.exists(log_path)
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "timestamp,seconds,steps,os,cpu,total_ram_mb,used_ram_mb,image_path\n"
            )

        steps_str = "" if num_steps is None else str(num_steps)
        total_ram_str = "" if total_ram_mb is None else f"{total_ram_mb:.1f}"
        used_ram_str = "" if used_ram_mb is None else f"{used_ram_mb:.1f}"

        f.write(
            f'"{timestamp}",{duration:.4f},{steps_str},'
            f'"{os_name}","{cpu_name}",{total_ram_str},{used_ram_str},"{image_path}"\n'
        )
