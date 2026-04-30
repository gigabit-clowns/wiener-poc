import numpy as np
import wiener_poc

def main():
    print("=== Rust PoC: Wiener Filter ===")
    
    in_data = np.ones(10, dtype=np.float32)
    print(f"Input: {in_data}")

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
