use crate::wiener_common::{compute_wiener_factors, HostMemoryAllocator, WienerParams};
use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(
    images: &[f32],
    batch: usize,
    n: usize,
    defocus: &[f32],
    params: &WienerParams,
    iterations: usize,
    allocator: HostMemoryAllocator,
) -> Result<Vec<f32>, DriverError> {
    let ctx = CudaContext::new(0)?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx.load_module(Ptx::from(ptx_src))?;

    let f_dft = module.load_function("dft2_forward_real_to_complex")?;
    let f_apply = module.load_function("apply_wiener_frequency")?;
    let f_idft = module.load_function("dft2_inverse_complex_to_real")?;

    let num_elements = batch * n * n;
    let num_freq = num_elements;
    let batch_i32 = batch as i32;
    let n_i32 = n as i32;

    let mut src_pinned = match allocator {
        HostMemoryAllocator::CudaHostAlloc => ctx.alloc_pinned::<f32>(num_elements)?,
    };
    let mut dst_pinned = match allocator {
        HostMemoryAllocator::CudaHostAlloc => ctx.alloc_pinned::<f32>(num_elements)?,
    };
    src_pinned.as_mut_slice()?.copy_from_slice(images);

    let stream_h2d = ctx.new_stream()?;
    let stream_comp = ctx.new_stream()?;
    let stream_d2h = ctx.new_stream()?;

    let event_uploaded = ctx.new_event(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)?;
    let event_computed = ctx.new_event(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT)?;

    let d_defocus = stream_comp.clone_htod(defocus)?;
    let wiener_factors = compute_wiener_factors(defocus, n, params);
    let d_wiener_factors = stream_comp.clone_htod(&wiener_factors)?;

    let threads = 128u32;
    let blocks_freq = (num_freq as u32 + threads - 1) / threads;
    let blocks_spatial = (num_elements as u32 + threads - 1) / threads;
    let cfg_freq = LaunchConfig {
        grid_dim: (blocks_freq, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    let cfg_spatial = LaunchConfig {
        grid_dim: (blocks_spatial, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let run_count = iterations.max(1);

    for _ in 0..run_count {
        let d_in = stream_h2d.clone_htod(&src_pinned)?;
        event_uploaded.record(&stream_h2d)?;
        event_uploaded.wait(&stream_comp)?;

        let mut d_freq_real = stream_comp.alloc_zeros::<f32>(num_freq)?;
        let mut d_freq_imag = stream_comp.alloc_zeros::<f32>(num_freq)?;
        let mut d_out = stream_comp.alloc_zeros::<f32>(num_elements)?;

        unsafe {
            stream_comp
                .launch_builder(&f_dft)
                .arg(&d_in)
                .arg(&mut d_freq_real)
                .arg(&mut d_freq_imag)
                .arg(&batch_i32)
                .arg(&n_i32)
                .launch(cfg_freq)?;

            stream_comp
                .launch_builder(&f_apply)
                .arg(&mut d_freq_real)
                .arg(&mut d_freq_imag)
                .arg(&d_defocus)
                .arg(&d_wiener_factors)
                .arg(&batch_i32)
                .arg(&n_i32)
                .arg(&params.pixel_size)
                .arg(&params.wavelength)
                .arg(&params.spherical_aberration)
                .arg(&params.q0)
                .launch(cfg_freq)?;

            stream_comp
                .launch_builder(&f_idft)
                .arg(&d_freq_real)
                .arg(&d_freq_imag)
                .arg(&mut d_out)
                .arg(&batch_i32)
                .arg(&n_i32)
                .launch(cfg_spatial)?;
        }

        event_computed.record(&stream_comp)?;
        event_computed.wait(&stream_d2h)?;
        stream_d2h.memcpy_dtoh(&d_out, &mut dst_pinned)?;
        stream_d2h.synchronize()?;

        if run_count > 1 {
            let last_out = dst_pinned.as_slice()?;
            src_pinned.as_mut_slice()?.copy_from_slice(last_out);
        }
    }

    Ok(dst_pinned.as_slice()?.to_vec())
}
