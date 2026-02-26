from optimum.onnxruntime import ORTStableDiffusionPipeline
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


MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
MODEL_NAME = MODEL_ID.split("/")[-1]


def main():
    ensure_directories()

    variant_name = f"{MODEL_NAME}_onnx"
    test_number = get_next_test_number(variant_name)
    test_tag = f"{test_number:04d}_{variant_name}"

    image_path = os.path.join(IMAGES_DIR, f"{test_tag}.png")
    log_path = os.path.join(LOGS_DIR, f"{test_tag}.log")

    start_total = time.perf_counter()

    # Load ORTStableDiffusionPipeline (exports to ONNX on first run)
    # Force CPUExecutionProvider to avoid GPU memory limitations / bad allocation errors
    t0 = time.perf_counter()
    ort_pipe = ORTStableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        export=True,
        provider="CPUExecutionProvider",
    )
    t1 = time.perf_counter()

    prompt = (
        "Cute and colorful birthday card featuring an adorable black cat. Vibrant digital "
        "illustration style, clean linework, centered composition, balloons and confetti in "
        "the background, soft lighting, cheerful colors, light background, greeting card aesthetic."
    )

    num_inference_steps = 4

    t2 = time.perf_counter()
    image = ort_pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=8.0,
        height=512,
        width=512,
        output_type="pil",
    ).images[0]
    t3 = time.perf_counter()

    t4 = time.perf_counter()
    image.save(image_path)
    t5 = time.perf_counter()

    end_total = time.perf_counter()

    timings = {
        "load_onnx_s": t1 - t0,
        "inference_latency_s": t3 - t2,
        "save_image_s": t5 - t4,
        "total_time_s": end_total - start_total,
    }

    specs = get_machine_specs()

    lines = build_base_report_lines(
        test_number=test_number,
        model_id=f"{MODEL_ID} (ORT/ONNX)",
        test_tag=test_tag,
        image_path=image_path,
        log_path=log_path,
        specs=specs,
    )
    lines.append("===== TIMINGS (SECONDS) =====")
    lines.append(f"Export/load ORT pipeline: {timings['load_onnx_s']:.3f}s")
    lines.append(f"Inference latency (ORT): {timings['inference_latency_s']:.3f}s")
    lines.append(f"Save image: {timings['save_image_s']:.3f}s")
    lines.append(f"Total time: {timings['total_time_s']:.3f}s")

    finalize_and_write_report(log_path, lines)


if __name__ == "__main__":
    main()
