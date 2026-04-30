fn main() {
    // This tells Cargo to recompile if we add or modify any CUDA code
    println!("cargo:rerun-if-changed=cuda_kernels/");
}
