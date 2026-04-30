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
    // alloc_pinned guarantees page-locked memory for fast DMA transfers.
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    src_pinned.copy_from_slice(data);

    // REQUIREMENT 5.2: Separate Streams
    let stream_h2d = ctx.new_stream()?;  // Stream A: Upload
    let stream_comp = ctx.new_stream()?; // Stream B: Compute
    let stream_d2h = ctx.new_stream()?;  // Stream C: Download

    // Events for synchronization
    let event_uploaded = ctx.new_event(None)?;
    let event_computed = ctx.new_event(None)?;

    // --- PIPELINE EXECUTION ---

    // STEP A: Upload data using Stream A
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    stream_h2d.record_event(&event_uploaded)?;

    // STEP B: Compute on Stream B
    // Wait for the upload event on Stream B
    stream_comp.wait_for_event(&event_uploaded)?;
    
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
    stream_comp.record_event(&event_computed)?;

    // STEP C: Download on Stream C
    // Wait for the computation event on Stream C
    stream_d2h.wait_for_event(&event_computed)?;
    
    // Download into Pinned Memory
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    // Final synchronization of the last stream
    stream_d2h.synchronize()?;

    // Return as a standard Vec
    Ok(dst_pinned.to_vec())
}
