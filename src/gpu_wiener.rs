use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    // 1. Setup Context and Module
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_kernel").unwrap();

    let num_elements = data.len();
    let n_i32 = num_elements as i32;

    // REQUIREMENT 5.1: Allocation with cudaHostAlloc (Pinned Memory)
    // We use alloc_pinned to guarantee page-locked memory for high-performance DMA.
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    // Use .as_mut_slice() to access the underlying memory of the PinnedHostSlice
    src_pinned.as_mut_slice().copy_from_slice(data);

    // REQUIREMENT 5.2: Separate Streams for Pipelining
    let stream_h2d = ctx.new_stream()?;  // Stream A: Upload
    let stream_comp = ctx.new_stream()?; // Stream B: Compute
    let stream_d2h = ctx.new_stream()?;  // Stream C: Download

    // Create events for GPU-side synchronization (Requirement 5.3)
    let event_uploaded = ctx.new_event(None)?;
    let event_computed = ctx.new_event(None)?;

    // --- PIPELINE EXECUTION ---

    // STEP A: Upload data using Stream A
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    
    // In cudarc 0.19.4, the event records itself into the stream
    event_uploaded.record(&stream_h2d)?;

    // STEP B: Compute on Stream B
    // Wait for Stream A to finish upload without blocking the CPU
    stream_comp.wait_event(&event_uploaded)?;
    
    let mut d_out = stream_comp.alloc_zeros::<f32>(num_elements)?;
    let threads = 256;
    let blocks = (num_elements as u32 + threads - 1) / threads;
    let cfg = LaunchConfig { 
        grid_dim: (blocks, 1, 1), 
        block_dim: (threads, 1, 1), 
        shared_mem_bytes: 0 
    };

    unsafe {
        stream_comp.launch_builder(&f)
            .arg(&mut d_out)
            .arg(&d_in)
            .arg(&n_i32)
            .launch(cfg)?;
    }
    
    // Record compute completion
    event_computed.record(&stream_comp)?;

    // STEP C: Download on Stream C
    // Wait for the kernel to finish
    stream_d2h.wait_event(&event_computed)?;
    
    // Download into Pinned Memory (Dst)
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    // Final synchronization: ensures the GPU has finished all work
    stream_d2h.synchronize()?;

    // Convert Pinned Memory back to Vec to return to Python
    Ok(dst_pinned.as_slice().to_vec())
}
