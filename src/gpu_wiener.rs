use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg, CudaStream, CudaEvent};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    // 1. Setup Context and Module
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_kernel").unwrap();

    let num_elements = data.len();
    let n_i32 = num_elements as i32;

    // REQUIREMENT 5.1: Allocation with Pinned Memory (cudaHostAlloc)
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    // We add '?' because as_mut_slice() now returns a Result
    src_pinned.as_mut_slice()?.copy_from_slice(data);

    // REQUIREMENT 5.2: Separate Streams
    let stream_h2d = ctx.new_stream()?;  // Stream A
    let stream_comp = ctx.new_stream()?; // Stream B
    let stream_d2h = ctx.new_stream()?;  // Stream C

    // REQUIREMENT 5.3: GPU Events for synchronization
    let event_uploaded = ctx.new_event(None)?;
    let event_computed = ctx.new_event(None)?;

    // --- PIPELINE EXECUTION ---

    // STEP A: Upload data using Stream A
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    
    // The event records its completion point in the stream
    event_uploaded.record(&stream_h2d)?;

    // STEP B: Compute on Stream B
    // The event instructs the stream to wait until it is reached
    event_uploaded.wait(&stream_comp)?;
    
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
    
    // Mark kernel completion
    event_computed.record(&stream_comp)?;

    // STEP C: Download on Stream C
    // Wait for Stream B to finish before starting download
    event_computed.wait(&stream_d2h)?;
    
    // Async copy into Pinned Memory
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    // 3. FINAL SYNCHRONIZATION
    stream_d2h.synchronize()?;

    // Return as a standard Vec (we use '?' on as_slice() too)
    Ok(dst_pinned.as_slice()?.to_vec())
}
