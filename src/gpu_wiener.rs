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

    // --- REQUISITO 5.1: RESERVA CON cudaHostAlloc (Pinned Memory) ---
    // Usamos alloc_host para garantizar memoria "page-locked".
    // Esto permite transferencias asíncronas reales por el bus PCIe.
    let mut src_pinned = ctx.alloc_host::<f32>(num_elements)?;
    let mut dst_pinned = ctx.alloc_host::<f32>(num_elements)?;

    // Copiamos los datos de entrada a nuestra memoria Pinned
    src_pinned.copy_from_slice(data);

    // 2. STREAMS & EVENTS (Requisito 5.2 - 5.4)
    let stream_h2d = ctx.new_stream()?;  // Stream A: Upload
    let stream_comp = ctx.new_stream()?; // Stream B: Compute
    let stream_d2h = ctx.new_stream()?;  // Stream C: Download

    let event_uploaded = ctx.create_event()?;
    let event_computed = ctx.create_event()?;

    // --- PIPELINE EXECUTION ---

    // STEP A: Upload data using Stream A
    // Al ser memoria Pinned, el DMA es directo y ultra rápido.
    let d_in = stream_h2d.clone_htod(&src_pinned)?;
    stream_h2d.record_event(&event_uploaded)?;

    // STEP B: Compute on Stream B
    // Espera por hardware a que el Stream A termine la subida
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
    stream_comp.record_event(&event_computed)?;

    // STEP C: Download on Stream C
    // Espera por hardware a que el Stream B termine el kernel
    stream_d2h.wait_event(&event_computed)?;
    
    // Descargamos directamente a nuestra memoria Pinned de destino
    stream_d2h.dtoh_copy_into(&d_out, &mut dst_pinned)?;

    // 3. FINAL SYNCHRONIZATION
    // Sincronizamos el stream de bajada para asegurar que dst_pinned es legible
    stream_d2h.synchronize()?;

    // Convertimos la memoria Pinned a un Vec estándar para devolverlo a Python
    // (En producción podríamos devolver la memoria Pinned directamente para evitar esta última copia)
    Ok(dst_pinned.to_vec())
}
