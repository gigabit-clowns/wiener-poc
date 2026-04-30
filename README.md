# Rust proof of concept: Wiener filter
## Introducción

Un filtro Wiener es un filtro que permite corregir la una función de transferencia (como la CTF) usando un enfoque de máxima verosimilitud.
Nótese que la CTF no es corregida del todo, sino que se hace “la mejor aproximación” que permita el nivel de señal a ruido (SNR)

El código en numpy de dicha función sería:
```py
import numpy as np


def _frequency2_grid_2d(box_size: int, pixel_size: float):
   kx = np.fft.rfftfreq(box_size, d=pixel_size)
   ky = np.fft.fftfreq(box_size, d=pixel_size)
   return np.square(kx[None,:]) + np.square(ky[:,None])


def _compute_ctf_image_2d(
   defocus: np.ndarray,
   box_size: int,
   pixel_size: float,
   wavelength: float,
   spherical_aberration: float,
   q0: float,
):
   k2 = _frequency2_grid_2d(
       box_size=box_size,
       pixel_size=pixel_size
   )
  
   wavelength2 = wavelength*wavelength
   angle = np.pi*wavelength*k2*(0.5*spherical_aberration*wavelength2*k2 + defocus[...,None,None])
   return np.sin(angle) - q0*np.cos(angle)


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
       q0=q0
   )
   images_ft = np.fft.rfft2(images)
   ctfs2 = np.square(ctfs)
   wiener_factor = 0.1 * np.mean(ctfs2, axis=(-1, -2), keepdims=True)
   wiener_corrected_images_ft = (images_ft * ctfs) / (ctfs2 + wiener_factor)
   return np.fft.irfft2(wiener_corrected_images_ft)
```

## Objetivo y Requisitos
Se quiere implementar esta función en Rust en un estilo orientado a rendimiento.
La implementación en Rust no necesita estar completa, ni ser funcional.
Pero si debe tener un roadmap **claro** y sin fisuras para abordar los siguientes requisitos.
No es válido copiar el código de numpy 1:1 con su equivalente de Rust ndarray, para eso está ya la versión de numpy.
No usamos numpy porque tenemos requisitos adicionales.
Los requisitos son flexibles siempre que se justifique que existe una alternativa que hace lo mismo.
El código que se haga no tiene que tener estándares altos de calidad, es una prueba de concepto.

- Soporte para CPU y GPU. En este caso se plantea que se pueda ejecutar en CPU y en CUDA como embajador de las GPUs.
- En CUDA se tienen que poder escribir kernels a mano. El proof of concept deberá tener al menos uno.
- En GPU se tiene poder ejecutar como un pipeline. Para ello, se requiere tener acceso a los Streams de CUDA o bien tener una capa por encima que lo gestione automáticamente (como sucede en JAX)
<img width="1796" height="524" alt="1" src="https://github.com/user-attachments/assets/89eb327b-2007-411d-8fd5-f769af8f1387" />

- Tiene que existir la posibilidad de usar un mecanismo de reserva de memoria custom (como por ejemplo uno que delegue en cudaHostAlloc)
- Para poner a prueba los últimos tres puntos se plantea la siguiente prueba:
1. Reservar 2 multidim-s (llamemosle SRC, y DST) en “host” con tamaño (B, N, N) donde B es el tamaño de batch y N es el tamaño de caja. El storage (contenido) de dichos arrays debe reservarse con cudaHostAlloc. 
2. Usar Stream A para subir los contenidos de SRC a una GPU
3. Steam B espera (cudaEvent) a que la transferencia del paso 2. acabe y lanza los kernels necesarios para que se aplique un filtro de Wiener. 
4. Steam C espera (cudaEvent) a que el cálculo del paso 3) acabe y copia el resultado a DST
5. Volver a paso 2.
- La implementación de CPU debe de poder soportar varias ISAs vectoriales (e.g. SSE2, AVX2, AVX512) en el mismo binario y determinar el más adecuado en tiempo de ejecución. Se permite usar frameworks equivalentes a Google Highway en C++. Si el compilador permite hacer esto de forma automática, es una opción más razonable. 
