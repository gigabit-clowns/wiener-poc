import numpy as np
import wiener_poc

def main():
    print("--- PoC: Rust + CUDA ---")

    # 1. Create a test array on CPU
    # Using float32 because or C++ kernel expects 'float*'
    in_data = np.ones(10, dtype=np.float32)
    print(f"Input (Host): {in_data}")

    # 2. We call our library (Rust orchestrates memory and runs PTX)
    out_data = wiener_poc.run_wiener_gpu(in_data)

    # 3. We verify result
    print(f"Output (Host): {out_data}")

    # As our dummy kernel multiplies by 0.1, we can validate it
    assert np.allclose(out_data, 0.1), "Error: GPU computation is not correct."
    print("Success! Pipeline is working from start to end.")

    print("-- Second test: Larger array --")
    in_data = np.ones(500_000_000, dtype=np.float32)
    try:
        while True:
            _ = wiener_poc.run_wiener_gpu(in_data)
    except KeyboardInterrupt:
        print("\nPrueba finalizada")

if __name__ == "__main__":
    main()
