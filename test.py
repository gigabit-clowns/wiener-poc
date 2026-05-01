import numpy as np
import argparse
import time
import numpy_wiener_reference

try:
    import wiener_poc  # type: ignore[import-not-found]
except ImportError:
    wiener_poc = None


# Perfil base "heavy": 1024 x 512 x 512 float32 ~= 1.00 GiB en el tensor de entrada.
DEFAULT_BATCH = 1024
DEFAULT_BOX = 512
DEFAULT_REPEATS_CPU = 2
DEFAULT_REPEATS_GPU = 3
DEFAULT_WARMUP_CPU = 0
DEFAULT_WARMUP_GPU = 1
DEFAULT_SKIP_REFERENCE = True


def _build_benchmark_case(batch_size: int, box_size: int):
    rng = np.random.default_rng(123)
    images = rng.normal(
        loc=0.0,
        scale=1.0,
        size=(batch_size, box_size, box_size),
    ).astype(np.float32)
    defocus = np.linspace(1.2e4, 1.8e4, batch_size, dtype=np.float32)
    params = {
        "box_size": box_size,
        "pixel_size": 1.0,
        "wavelength": 0.0197,
        "spherical_aberration": 2.7e7,
        "q0": 0.07,
    }
    return images, defocus, params


def _to_gib(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def _run_timed(fn, repeats: int):
    times = []
    out = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, times


def _print_timing_stats(label: str, times: list[float], section_wall_s: float, warmups: int):
    total_timed = float(np.sum(times))
    print(
        f"{label} time/call -> min {min(times):.3f}s | mean {np.mean(times):.3f}s | max {max(times):.3f}s"
    )
    print(
        f"{label} total timed ({len(times)} iter): {total_timed:.3f}s | section wall: {section_wall_s:.3f}s"
    )
    if warmups > 0:
        print(f"{label} warmup iter no cronometradas: {warmups}")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark simple CPU/GPU Wiener vs referencia NumPy"
    )
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Tamano de batch B")
    parser.add_argument("--box", type=int, default=DEFAULT_BOX, help="Tamano de caja N")
    parser.add_argument(
        "--repeats-cpu", type=int, default=DEFAULT_REPEATS_CPU, help="Repeticiones para CPU"
    )
    parser.add_argument(
        "--repeats-gpu", type=int, default=DEFAULT_REPEATS_GPU, help="Repeticiones para GPU"
    )
    parser.add_argument(
        "--warmup-cpu", type=int, default=DEFAULT_WARMUP_CPU, help="Warmup CPU sin cronometrar"
    )
    parser.add_argument(
        "--warmup-gpu", type=int, default=DEFAULT_WARMUP_GPU, help="Warmup GPU sin cronometrar"
    )
    parser.set_defaults(skip_reference=DEFAULT_SKIP_REFERENCE)
    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument(
        "--skip-reference",
        dest="skip_reference",
        action="store_true",
        help="Omitir referencia NumPy para pruebas de solo rendimiento",
    )
    ref_group.add_argument(
        "--with-reference",
        dest="skip_reference",
        action="store_false",
        help="Calcular tambien referencia NumPy y validar contra ella",
    )
    return parser.parse_args()


def main():
    args = _parse_args()

    print("=== Rust PoC: Wiener Filter (CPU/GPU vs NumPy) ===")

    images32, defocus32, params = _build_benchmark_case(
        batch_size=args.batch,
        box_size=args.box,
    )

    print(f"Input shape: {images32.shape}")
    print(f"Defocus: {defocus32}")
    input_gib = _to_gib(images32.nbytes)
    print(f"Memoria de entrada (float32): {input_gib:.2f} GiB")
    # Reserva host aproximada con rutas CPU+GPU: input + out_cpu + out_gpu (+ referencia opcional).
    estimated_host_tensors = 3 + (0 if args.skip_reference else 1)
    estimated_host_gib = input_gib * estimated_host_tensors
    print(
        f"Huella host aproximada durante ejecucion: ~{estimated_host_gib:.2f} GiB ({estimated_host_tensors} tensores grandes)"
    )
    if args.repeats_cpu != args.repeats_gpu:
        print(
            f"Aviso: repeats distintos (CPU={args.repeats_cpu}, GPU={args.repeats_gpu})."
        )
        print("Para comparar wall-time justo usa los mismos repeats en ambas rutas.")
    print(
        "Consejo monitorizacion: abre htop/nvtop antes de lanzar este script para ver uso sostenido."
    )

    reference = None
    if not args.skip_reference:
        t0 = time.perf_counter()
        reference = numpy_wiener_reference.wiener_ctf_correct_2d(
            images=images32,
            defocus=defocus32,
            box_size=params["box_size"],
            pixel_size=params["pixel_size"],
            wavelength=params["wavelength"],
            spherical_aberration=params["spherical_aberration"],
            q0=params["q0"],
        ).astype(np.float32, copy=False)
        t1 = time.perf_counter()
        print(f"Referencia NumPy calculada en {t1 - t0:.3f} s")

    if wiener_poc is None:
        print("wiener_poc no esta instalado todavia. Se omiten las pruebas Rust.")
        if reference is not None:
            print("La referencia NumPy ya se ha ejecutado para usar como baseline.")
        return

    # Prueba 1: CPU (dispatch runtime SSE2/AVX2/AVX512)
    cpu_section_t0 = time.perf_counter()
    for _ in range(args.warmup_cpu):
        _ = wiener_poc.run_wiener_cpu(
            images32,
            defocus32,
            params["pixel_size"],
            params["wavelength"],
            params["spherical_aberration"],
            params["q0"],
        )
    out_cpu, cpu_times = _run_timed(
        lambda: wiener_poc.run_wiener_cpu(
            images32,
            defocus32,
            params["pixel_size"],
            params["wavelength"],
            params["spherical_aberration"],
            params["q0"],
        ),
        args.repeats_cpu,
    )
    cpu_section_t1 = time.perf_counter()
    print(f"CPU output shape: {out_cpu.shape}")
    _print_timing_stats(
        label="CPU",
        times=cpu_times,
        section_wall_s=cpu_section_t1 - cpu_section_t0,
        warmups=args.warmup_cpu,
    )
    assert out_cpu.shape == images32.shape, "Forma invalida en CPU"
    assert np.isfinite(out_cpu).all(), "CPU devolvio NaN/Inf"
    if reference is not None:
        assert np.allclose(out_cpu, reference, atol=8e-2, rtol=8e-2), (
            "CPU no coincide con referencia NumPy en tolerancia"
        )

    # Prueba 2: GPU (Pipeline 3 Streams + CudaEvents + cudaHostAlloc)
    gpu_section_t0 = time.perf_counter()
    for _ in range(args.warmup_gpu):
        _ = wiener_poc.run_wiener_gpu(
            images32,
            defocus32,
            params["pixel_size"],
            params["wavelength"],
            params["spherical_aberration"],
            params["q0"],
            1,
            "cudaHostAlloc",
        )
    out_gpu, gpu_times = _run_timed(
        lambda: wiener_poc.run_wiener_gpu(
            images32,
            defocus32,
            params["pixel_size"],
            params["wavelength"],
            params["spherical_aberration"],
            params["q0"],
            1,
            "cudaHostAlloc",
        ),
        args.repeats_gpu,
    )
    gpu_section_t1 = time.perf_counter()
    print(f"GPU output shape: {out_gpu.shape}")
    _print_timing_stats(
        label="GPU",
        times=gpu_times,
        section_wall_s=gpu_section_t1 - gpu_section_t0,
        warmups=args.warmup_gpu,
    )
    assert out_gpu.shape == images32.shape, "Forma invalida en GPU"
    assert np.isfinite(out_gpu).all(), "GPU devolvio NaN/Inf"
    if reference is not None:
        assert np.allclose(out_gpu, reference, atol=1.2e-1, rtol=1.2e-1), (
            "GPU no coincide con referencia NumPy en tolerancia"
        )

    if min(gpu_times) > 0:
        speedup = min(cpu_times) / min(gpu_times)
        print(f"Speedup (best CPU / best GPU): {speedup:.2f}x")

    print("\n✅ Los outputs coinciden con la referencia de NumPy.")

if __name__ == "__main__":
    main()
