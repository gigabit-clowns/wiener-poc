import numpy as np


def _frequency2_grid_2d(box_size: int, pixel_size: float) -> np.ndarray:
    kx = np.fft.rfftfreq(box_size, d=pixel_size)
    ky = np.fft.fftfreq(box_size, d=pixel_size)
    return np.square(kx[None, :]) + np.square(ky[:, None])


def _compute_ctf_image_2d(
    defocus: np.ndarray,
    box_size: int,
    pixel_size: float,
    wavelength: float,
    spherical_aberration: float,
    q0: float,
) -> np.ndarray:
    k2 = _frequency2_grid_2d(box_size=box_size, pixel_size=pixel_size)

    wavelength2 = wavelength * wavelength
    angle = np.pi * wavelength * k2 * (
        0.5 * spherical_aberration * wavelength2 * k2 + defocus[..., None, None]
    )
    return np.sin(angle) - q0 * np.cos(angle)


def wiener_ctf_correct_2d(
    images: np.ndarray,
    defocus: np.ndarray,
    box_size: int,
    pixel_size: float,
    wavelength: float,
    spherical_aberration: float,
    q0: float,
) -> np.ndarray:
    ctfs = _compute_ctf_image_2d(
        defocus=defocus,
        box_size=box_size,
        pixel_size=pixel_size,
        wavelength=wavelength,
        spherical_aberration=spherical_aberration,
        q0=q0,
    )
    images_ft = np.fft.rfft2(images)
    ctfs2 = np.square(ctfs)
    wiener_factor = 0.1 * np.mean(ctfs2, axis=(-1, -2), keepdims=True)
    wiener_corrected_images_ft = (images_ft * ctfs) / (ctfs2 + wiener_factor)
    return np.fft.irfft2(wiener_corrected_images_ft, s=(box_size, box_size))


def build_test_case(batch_size: int = 2, box_size: int = 8):
    rng = np.random.default_rng(123)
    images = rng.normal(loc=0.0, scale=1.0, size=(batch_size, box_size, box_size))

    # Use a physically plausible defocus range in Angstroms.
    defocus = np.linspace(1.2e4, 1.8e4, batch_size, dtype=np.float64)

    params = {
        "box_size": box_size,
        "pixel_size": 1.0,
        "wavelength": 0.0197,
        "spherical_aberration": 2.7e7,
        "q0": 0.07,
    }
    return images.astype(np.float64), defocus, params


def main() -> None:
    images, defocus, params = build_test_case(batch_size=2, box_size=8)

    corrected = wiener_ctf_correct_2d(
        images=images,
        defocus=defocus,
        box_size=params["box_size"],
        pixel_size=params["pixel_size"],
        wavelength=params["wavelength"],
        spherical_aberration=params["spherical_aberration"],
        q0=params["q0"],
    )

    print("=== NumPy Wiener Reference ===")
    print(f"Input shape: {images.shape}, dtype: {images.dtype}")
    print(f"Defocus: {defocus}")
    print(f"Output shape: {corrected.shape}, dtype: {corrected.dtype}")
    print("First image, top-left 3x3 output block:")
    print(np.array2string(corrected[0, :3, :3], precision=6))

    assert corrected.shape == images.shape
    assert np.isfinite(corrected).all()
    print("\nReference run OK.")


if __name__ == "__main__":
    main()
