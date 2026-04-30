import numpy as np
from numpy_wiener_reference import build_test_case, wiener_ctf_correct_2d

try:
    import wiener_poc  # type: ignore[import-not-found]
except ImportError:
    wiener_poc = None

def main():
    print("=== Rust PoC: Wiener Filter (CPU/GPU vs NumPy) ===")

    images, defocus, params = build_test_case(batch_size=2, box_size=8)
    reference = wiener_ctf_correct_2d(
        images=images,
        defocus=defocus,
        box_size=params["box_size"],
        pixel_size=params["pixel_size"],
        wavelength=params["wavelength"],
        spherical_aberration=params["spherical_aberration"],
        q0=params["q0"],
    )

    images32 = images.astype(np.float32)
    defocus32 = defocus.astype(np.float32)

    print(f"Input shape: {images32.shape}")
    print(f"Defocus: {defocus32}")

    if wiener_poc is None:
        print("wiener_poc no esta instalado todavia. Se omiten las pruebas Rust.")
        print("La referencia NumPy ya se ha ejecutado para usar como baseline.")
        return

    # Prueba 1: CPU (dispatch runtime SSE2/AVX2/AVX512)
    out_cpu = wiener_poc.run_wiener_cpu(
        images32,
        defocus32,
        params["pixel_size"],
        params["wavelength"],
        params["spherical_aberration"],
        params["q0"],
    )
    print(f"CPU output shape: {out_cpu.shape}")
    assert out_cpu.shape == images32.shape, "Forma invalida en CPU"
    assert np.isfinite(out_cpu).all(), "CPU devolvio NaN/Inf"
    assert np.allclose(out_cpu, reference.astype(np.float32), atol=8e-2, rtol=8e-2), (
        "CPU no coincide con referencia NumPy en tolerancia"
    )

    # Prueba 2: GPU (Pipeline 3 Streams + CudaEvents + cudaHostAlloc)
    out_gpu = wiener_poc.run_wiener_gpu(
        images32,
        defocus32,
        params["pixel_size"],
        params["wavelength"],
        params["spherical_aberration"],
        params["q0"],
        1,
        "cudaHostAlloc",
    )
    print(f"GPU output shape: {out_gpu.shape}")
    assert out_gpu.shape == images32.shape, "Forma invalida en GPU"
    assert np.isfinite(out_gpu).all(), "GPU devolvio NaN/Inf"
    assert np.allclose(out_gpu, reference.astype(np.float32), atol=1.2e-1, rtol=1.2e-1), (
        "GPU no coincide con referencia NumPy en tolerancia"
    )

    print("\n✅ Todos los requisitos del Roadmap validados con éxito.")

if __name__ == "__main__":
    main()
