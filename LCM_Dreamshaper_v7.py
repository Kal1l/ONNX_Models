from diffusers import DiffusionPipeline
import os
import torch
import time
from report_utils import (
    LOGS_DIR,
    IMAGES_DIR,
    ensure_directories,
    get_next_test_number,
    get_machine_specs,
    build_base_report_lines,
    finalize_and_write_report,
)


MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
MODEL_NAME = MODEL_ID.split("/")[-1]


def main():
    ensure_directories()

    test_number = get_next_test_number(MODEL_NAME)
    test_tag = f"{test_number:04d}_{MODEL_NAME}"

    image_path = os.path.join(IMAGES_DIR, f"{test_tag}.png")
    log_path = os.path.join(LOGS_DIR, f"{test_tag}.log")

    start_total = time.perf_counter()

    # Load the model locally (downloads only the first time)
    t0 = time.perf_counter()
    pipe = DiffusionPipeline.from_pretrained("SimianLuo/LCM_Dreamshaper_v7")
    t1 = time.perf_counter()

    t2 = time.perf_counter()
    pipe.to(torch_device="cpu", torch_dtype=torch.float32)
    t3 = time.perf_counter()

    # Birthday card preset with the black cat
    prompt = (
        "Cute and colorful birthday card featuring an adorable black cat. Vibrant digital illustration style, clean linework, centered composition, balloons and confetti in the background, soft lighting, cheerful colors, light background, greeting card aesthetic."
    )

    # LCM works very well with few steps
    num_inference_steps = 4  # recommended: 4–6

    t4 = time.perf_counter()
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.0,
        lcm_origin_steps=50,
        output_type="pil",
    ).images[0]
    t5 = time.perf_counter()

    t6 = time.perf_counter()
    image.save(image_path)
    t7 = time.perf_counter()

    end_total = time.perf_counter()

    timings = {
        "load_model_s": t1 - t0,
        "move_to_device_s": t3 - t2,
        "inference_latency_s": t5 - t4,
        "save_image_s": t7 - t6,
        "total_time_s": end_total - start_total,
    }

    specs = get_machine_specs()

    lines = build_base_report_lines(
        test_number=test_number,
        model_id=MODEL_ID,
        test_tag=test_tag,
        image_path=image_path,
        log_path=log_path,
        specs=specs,
    )
    lines.append("===== TIMINGS (SECONDS) =====")
    lines.append(f"Load model: {timings['load_model_s']:.3f}s")
    lines.append(f"Move to device: {timings['move_to_device_s']:.3f}s")
    lines.append(f"Inference latency (pipe): {timings['inference_latency_s']:.3f}s")
    lines.append(f"Save image: {timings['save_image_s']:.3f}s")
    lines.append(f"Total time: {timings['total_time_s']:.3f}s")

    finalize_and_write_report(log_path, lines)


if __name__ == "__main__":
    main()