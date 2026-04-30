use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // This tells Cargo to recompile if we add or modify any CUDA code
    println!("cargo:rerun-if-changed=cuda_kernels/wiener.cu");

    // Rust provides a temp folder for build artifacts
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let ptx_path = out_dir.join("wiener.ptx");

    // Invoke nvcc to generate PTX
    let status = Command::new("nvcc")
        .arg("-ptx") // Generate intermediate PTX code
        .arg("-arch=compute_61") // Defines architecture to be built to
        .arg("cuda_kernels/wiener.cu")
        .arg("-o")
        .arg(&ptx_path)
        .status()
        .expect("Error running nvcc. Is CUDA in PATH?");

    assert!(status.success(), "Error compiling CUDA kernel. Check syntax in wiener.cu");
}
