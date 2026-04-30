use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

pub fn execute_wiener_pipeline_events(data: &[f32]) -> Result<Vec<f32>, DriverError> {
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;
    let f = module.load_function("wiener_kernel").unwrap();

    let num_elements = data.len();
    let n_i32 = num_elements as i32;

    // 1. Pinned Memory (Usa alloc_pinned de CudaContext)
    let mut src_pinned = ctx.alloc_pinned::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_pinned::<f32>(num_elements)?;

    // .as_mut_slice() devuelve un Result, requiere '?'
    src_pinned.as_mut_slice()?.copy_from_slice(data);

    // 2. Streams & Events
    let stream_h2d = ctx.new_stream()?;  
    let stream_comp = ctx.new_stream()?; 
    let stream_d2h = ctx.new_stream()?;  

    // new_event requiere flags de la capa sys. 0 es el valor por defecto (CU_EVENT_DEFAULT)
    let event_uploaded = ctx.new_event(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)?;
    let event_computed = ctx.new_event(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)?;

    // --- PIPELINE EXECUTION ---

    // A. Upload (Stream A)
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    
    // El método está en CudaEvent: record(&self, stream: &Arc<CudaStream>)
    event_uploaded.record(&stream_h2d)?;

    // B. Compute (Stream B)
    // El método está en CudaEvent: wait(&self, stream: &Arc<CudaStream>)
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
    
    event_computed.record(&stream_comp)?;

    // C. Download (Stream C)
    event_computed.wait(&stream_d2h)?;
    
    // memcpy_dtoh copia desde el Device a un PinnedHostSlice asíncronamente
    stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;

    // Sincronizamos el stream final para asegurar que los datos están en el Host
    stream_d2h.synchronize()?;

    // Devolvemos el resultado (as_slice() devuelve un Result)
    Ok(dst_pinned.as_slice()?.to_vec())
}
