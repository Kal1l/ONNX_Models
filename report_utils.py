import os
import platform
import re
import torch
from typing import Dict, List


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR = os.path.join(BASE_DIR, "logs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")


def ensure_directories() -> None:
    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def get_next_test_number(model_name: str) -> int:
    """Descobre o próximo número de teste baseado em arquivos no diretório de logs."""
    pattern = re.compile(rf"^(\d+)_" + re.escape(model_name) + r"\.")
    highest = 0
    if not os.path.isdir(LOGS_DIR):
        return 1

    for filename in os.listdir(LOGS_DIR):
        match = pattern.match(filename)
        if match:
            try:
                num = int(match.group(1))
                if num > highest:
                    highest = num
            except ValueError:
                continue

    return highest + 1 if highest > 0 else 1


def get_machine_specs() -> Dict[str, object]:
    uname = platform.uname()
    processor = platform.processor() or uname.processor

    specs: Dict[str, object] = {
        "os": f"{uname.system} {uname.release}",
        "os_version": uname.version,
        "python_version": platform.python_version(),
        "processor": processor,
        "cpu_cores": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
    }

    if torch.cuda.is_available():
        try:
            specs["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            specs["gpu_name"] = "Unknown CUDA GPU"

    return specs


def build_base_report_lines(
    test_number: int,
    model_id: str,
    test_tag: str,
    image_path: str,
    log_path: str,
    specs: Dict[str, object],
) -> List[str]:
    """Monta as linhas comuns de cabeçalho do relatório (sem timings)."""
    lines: List[str] = []
    lines.append(f"Test: {test_number}")
    lines.append(f"Model: {model_id}")
    lines.append(f"Test tag: {test_tag}")
    lines.append("")
    lines.append(f"Image saved at: {image_path}")
    lines.append(f"Log file: {log_path}")
    lines.append("")
    lines.append("===== MACHINE SPECS =====")
    lines.append(f"Operating System: {specs['os']} ({specs['os_version']})")
    lines.append(f"Python: {specs['python_version']}")
    lines.append(f"Processor: {specs['processor']}")
    lines.append(f"CPU cores (logical): {specs['cpu_cores']}")
    lines.append(f"CUDA available: {specs['cuda_available']}")
    if specs.get("gpu_name"):
        lines.append(f"GPU: {specs['gpu_name']}")
    lines.append("")
    return lines


def finalize_and_write_report(log_path: str, lines: List[str]) -> None:
    """Converte as linhas em texto, imprime e grava no arquivo de log."""
    report = "\n".join(lines)
    print(report)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
