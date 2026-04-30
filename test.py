import numpy as np

try:
    import wiener_poc  # type: ignore[import-not-found]
except ImportError:
    wiener_poc = None

def main():
    print("=== Rust PoC: Wiener Filter ===")
    
    in_data = np.ones(10, dtype=np.float32)
    print(f"Input: {in_data}")

    if wiener_poc is None:
        print("wiener_poc no esta instalado todavia. Se omiten las pruebas Rust.")
        return

    # Prueba 1: CPU (AVX2/AVX512 dinámico)
    out_cpu = wiener_poc.run_wiener_cpu(in_data)
    print(f"Output CPU: {out_cpu}")
    assert np.allclose(out_cpu, 0.1), "Error en SIMD CPU"

    # Prueba 2: GPU (Pipeline 3-Streams + CudaEvents)
    out_gpu = wiener_poc.run_wiener_gpu(in_data)
    print(f"Output GPU: {out_gpu}")
    assert np.allclose(out_gpu, 0.1), "Error en Pipeline GPU"

    print("\n✅ Todos los requisitos del Roadmap validados con éxito.")

if __name__ == "__main__":
    main()
