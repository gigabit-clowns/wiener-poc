use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_kernel").unwrap();

    let num_elements = data.len();
    let n_i32 = num_elements as i32;

    // 1. Pinned Memory (Requisito 5.1)
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    // Usamos '?' porque as_mut_slice() devuelve un Result
    src_pinned.as_mut_slice()?.copy_from_slice(data);

    // 2. Streams & Events
    let stream_h2d = ctx.new_stream()?;  
    let stream_comp = ctx.new_stream()?; 
    let stream_d2h = ctx.new_stream()?;  

    let event_uploaded = ctx.new_event(None)?;
    let event_computed = ctx.new_event(None)?;

    // A. Upload
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    event_uploaded.record(&stream_h2d)?; // El evento se graba en el stream

    // B. Compute 
    // ¡OJO! El método es wait_for_event
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
    event_computed.record(&stream_comp)?;

    // C. Download
    // ¡OJO! Aquí también: wait_for_event
    stream_d2h.wait_for_event(&event_computed)?;
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    stream_d2h.synchronize()?;

    // Devolvemos el resultado usando as_slice()?
    Ok(dst_pinned.as_slice()?.to_vec())
}
