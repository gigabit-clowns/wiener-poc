use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::mpsc;
use std::thread;

pub fn execute_wiener_pipeline(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    // 1. Initialize context and load the kernel
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_dummy").unwrap();

    let num_elements = data.len();
    let num_chunks = 4; // Divide the data into 4 chunks
    let chunk_size = num_elements / num_chunks; 
    
    // Final array where we will store the processed data
    let mut final_result = vec![0.0f32; num_elements];

    // Channels to pass GPU memory pointers between threads
    let (tx_comp, rx_comp) = mpsc::channel();
    let (tx_d2h, rx_d2h) = mpsc::channel();

    // Clone the CudaContext (which is an Arc internally) for each thread
    // This is cheap and allows each thread to own a reference to the GPU context.
    let ctx_h2d = ctx.clone();
    let ctx_comp = ctx.clone();
    let ctx_d2h = ctx.clone();

    // Create a mutable reference to safely pass into Thread 3
    let result_ref = &mut final_result;

    // 2. FEARLESS CONCURRENCY: Create a thread "Scope". 
    thread::scope(|s| {
        // ---------------------------------------------------------
        // THREAD 1: Upload (Host -> Device)
        // ---------------------------------------------------------
        s.spawn(move || {
            let stream_h2d = ctx_h2d.new_stream().unwrap();
            for i in 0..num_chunks {
                let start = i * chunk_size;
                // Adjust in case the last chunk is not perfectly divisible
                let end = if i == num_chunks - 1 { num_elements } else { start + chunk_size };
                
                let chunk_data = &data[start..end];
                
                // clone_htod uploads data to VRAM on this stream
                let d_in = stream_h2d.clone_htod(chunk_data).unwrap();
                
                // Send the GPU memory chunk to the compute thread
                tx_comp.send((start, end, d_in)).unwrap();
            }
        });

        // ---------------------------------------------------------
        // THREAD 2: Compute (JIT Kernel)
        // ---------------------------------------------------------
        s.spawn(move || {
            let stream_comp = ctx_comp.new_stream().unwrap();
            
            // This loop sleeps until data arrives from Thread 1
            for (start, end, d_in) in rx_comp {
                let size = end - start;
                let mut d_out = stream_comp.alloc_zeros::<f32>(size).unwrap();
                let n_i32 = size as i32;

                let threads = 256;
                let blocks = (size as u32 + threads - 1) / threads;
                let cfg = LaunchConfig {
                    grid_dim: (blocks, 1, 1),
                    block_dim: (threads, 1, 1),
                    shared_mem_bytes: 0,
                };

                // Launch the computation
                unsafe { 
                    stream_comp.launch_builder(&f)
                        .arg(&mut d_out)
                        .arg(&d_in)
                        .arg(&n_i32)
                        .launch(cfg) 
                }.unwrap();

                // Synchronize ONLY this stream to avoid passing garbage results to Thread 3
                stream_comp.synchronize().unwrap();
                
                // Send the result (on GPU) to Thread 3
                tx_d2h.send((start, end, d_out)).unwrap();
            }
        });

        // ---------------------------------------------------------
        // THREAD 3: Download (Device -> Host)
        // ---------------------------------------------------------
        s.spawn(move || {
            let stream_d2h = ctx_d2h.new_stream().unwrap();
            
            // This loop sleeps until Thread 2 passes the processed chunk
            for (start, end, d_out) in rx_d2h {
                // Download back to the CPU
                let host_chunk = stream_d2h.clone_dtoh(&d_out).unwrap();
                
                // Write directly into the final memory. 
                result_ref[start..end].copy_from_slice(&host_chunk);
            }
        });
    });

    Ok(final_result)
}
