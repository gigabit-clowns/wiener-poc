use crate::wiener_common::{compute_wiener_factors, HostMemoryAllocator, WienerParams};
use cudarc::cufft::{sys::cufftType, sys::float2, CudaFft};
use cudarc::driver::{CudaContext, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

pub fn execute_wiener_pipeline_events(
    images: &[f32],
    batch: usize,
    n: usize,
    defocus: &[f32],
    params: &WienerParams,
    iterations: usize,
    allocator: HostMemoryAllocator,
) -> Result<Vec<f32>, String> {
    if images.len() != batch * n * n {
        return Err(format!(
            "Invalid image buffer size. Got {}, expected {}",
            images.len(),
            batch * n * n
        ));
    }

    if defocus.len() != batch {
        return Err(format!(
            "Invalid defocus length. Got {}, expected {}",
            defocus.len(),
            batch
        ));
    }

    let ctx = CudaContext::new(0).map_err(|e| format!("{e:?}"))?;
    let ptx_src = include_str!(concat!(env!("OUT_DIR"), "/wiener.ptx"));
    let module = ctx
        .load_module(Ptx::from(ptx_src))
        .map_err(|e| format!("{e:?}"))?;

    let f_apply = module
        .load_function("apply_wiener_frequency")
        .map_err(|e| format!("{e:?}"))?;
    let f_norm = module
        .load_function("normalize_real_image")
        .map_err(|e| format!("{e:?}"))?;

    let num_elements = batch * n * n;
    let freq_width = n / 2 + 1;
    let num_freq = batch * n * freq_width;
    let batch_i32 = batch as i32;
    let n_i32 = n as i32;
    let freq_width_i32 = freq_width as i32;
    let num_elements_i32 = num_elements as i32;
    let norm = 1.0f32 / ((n * n) as f32);

    let mut src_pinned = match allocator {
        HostMemoryAllocator::CudaHostAlloc => ctx
            .alloc_pinned::<f32>(num_elements)
            .map_err(|e| format!("{e:?}"))?,
    };
    let mut dst_pinned = match allocator {
        HostMemoryAllocator::CudaHostAlloc => ctx
            .alloc_pinned::<f32>(num_elements)
            .map_err(|e| format!("{e:?}"))?,
    };
    src_pinned
        .as_mut_slice()
        .map_err(|e| format!("{e:?}"))?
        .copy_from_slice(images);

    let stream_h2d = ctx.new_stream().map_err(|e| format!("{e:?}"))?;
    let stream_comp = ctx.new_stream().map_err(|e| format!("{e:?}"))?;
    let stream_d2h = ctx.new_stream().map_err(|e| format!("{e:?}"))?;

    let event_uploaded = ctx
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("{e:?}"))?;
    let event_computed = ctx
        .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(|e| format!("{e:?}"))?;

    let d_defocus = stream_comp
        .clone_htod(defocus)
        .map_err(|e| format!("{e:?}"))?;
    let wiener_factors = compute_wiener_factors(defocus, n, params);
    let d_wiener_factors = stream_comp
        .clone_htod(&wiener_factors)
        .map_err(|e| format!("{e:?}"))?;

    let rank = [n_i32, n_i32];
    let out_embed = [n_i32, freq_width_i32];
    let in_embed = [n_i32, freq_width_i32];

    let fft_r2c = CudaFft::plan_many(
        &rank,
        None,
        1,
        (n * n) as i32,
        Some(&out_embed),
        1,
        (n * freq_width) as i32,
        cufftType::CUFFT_R2C,
        batch_i32,
        stream_comp.clone(),
    )
    .map_err(|e| format!("{e:?}"))?;

    let fft_c2r = CudaFft::plan_many(
        &rank,
        Some(&in_embed),
        1,
        (n * freq_width) as i32,
        None,
        1,
        (n * n) as i32,
        cufftType::CUFFT_C2R,
        batch_i32,
        stream_comp.clone(),
    )
    .map_err(|e| format!("{e:?}"))?;

    let mut d_in = stream_h2d
        .alloc_zeros::<f32>(num_elements)
        .map_err(|e| format!("{e:?}"))?;
    let mut d_freq = stream_comp
        .alloc_zeros::<float2>(num_freq)
        .map_err(|e| format!("{e:?}"))?;
    let mut d_out = stream_comp
        .alloc_zeros::<f32>(num_elements)
        .map_err(|e| format!("{e:?}"))?;

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
        stream_h2d
            .memcpy_htod(&src_pinned, &mut d_in)
            .map_err(|e| format!("{e:?}"))?;
        event_uploaded
            .record(&stream_h2d)
            .map_err(|e| format!("{e:?}"))?;
        stream_comp
            .wait(&event_uploaded)
            .map_err(|e| format!("{e:?}"))?;

        fft_r2c
            .exec_r2c(&d_in, &mut d_freq)
            .map_err(|e| format!("{e:?}"))?;

        unsafe {
            stream_comp
                .launch_builder(&f_apply)
                .arg(&mut d_freq)
                .arg(&d_defocus)
                .arg(&d_wiener_factors)
                .arg(&batch_i32)
                .arg(&n_i32)
                .arg(&freq_width_i32)
                .arg(&params.pixel_size)
                .arg(&params.wavelength)
                .arg(&params.spherical_aberration)
                .arg(&params.q0)
                .launch(cfg_freq)
                .map_err(|e| format!("{e:?}"))?;
        }

        fft_c2r
            .exec_c2r(&mut d_freq, &mut d_out)
            .map_err(|e| format!("{e:?}"))?;

        unsafe {
            stream_comp
                .launch_builder(&f_norm)
                .arg(&mut d_out)
                .arg(&num_elements_i32)
                .arg(&norm)
                .launch(cfg_spatial)
                .map_err(|e| format!("{e:?}"))?;
        }

        event_computed
            .record(&stream_comp)
            .map_err(|e| format!("{e:?}"))?;
        stream_d2h
            .wait(&event_computed)
            .map_err(|e| format!("{e:?}"))?;
        stream_d2h
            .memcpy_dtoh(&d_out, &mut dst_pinned)
            .map_err(|e| format!("{e:?}"))?;
        stream_d2h.synchronize().map_err(|e| format!("{e:?}"))?;

        if run_count > 1 {
            let last_out = dst_pinned.as_slice().map_err(|e| format!("{e:?}"))?;
            src_pinned
                .as_mut_slice()
                .map_err(|e| format!("{e:?}"))?
                .copy_from_slice(last_out);
        }
    }

    Ok(dst_pinned
        .as_slice()
        .map_err(|e| format!("{e:?}"))?
        .to_vec())
}
