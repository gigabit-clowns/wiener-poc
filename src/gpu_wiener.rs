use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    // 1. Setup
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_kernel").unwrap();

    let num_elements = data.len();
    let n_i32 = num_elements as i32;

    // 2. Pinned Memory (Requirement 5.1)
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    // as_mut_slice() devuelve un Result<&mut [T], DriverError>, necesitamos el '?'
    src_pinned.as_mut_slice()?.copy_from_slice(data);

    // 3. Streams & Events (Requirement 5.2)
    let stream_h2d = ctx.new_stream()?;  
    let stream_comp = ctx.new_stream()?; 
    let stream_d2h = ctx.new_stream()?;  

    // En 0.19.4 new_event requiere flags. Usamos Default (0).
    let event_uploaded = ctx.new_event(Default::default())?;
    let event_computed = ctx.new_event(Default::default())?;

    // --- PIPELINE EXECUTION ---

    // STEP A: Upload
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    
    // IMPORTANTE: En cudarc 0.19.4 la sincronización se hace a través del CONTEXTO
    ctx.record_event(&stream_h2d, &event_uploaded)?;

    // STEP B: Compute
    // El contexto ordena a stream_comp esperar por event_uploaded
    ctx.wait_event(&stream_comp, &event_uploaded)?;
    
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
    
    ctx.record_event(&stream_comp, &event_computed)?;

    // STEP C: Download
    ctx.wait_event(&stream_d2h, &event_computed)?;
    
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    // Sincronizamos para poder leer los datos en el Host
    stream_d2h.synchronize()?;

    // as_slice() también devuelve un Result
    Ok(dst_pinned.as_slice()?.to_vec())
}
