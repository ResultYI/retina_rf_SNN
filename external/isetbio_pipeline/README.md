# ISETBio cone-response export

This adapter converts a calibrated RGB video, frame directory, or still image
into the native cone-mosaic response used by the retina-inspired SNN.

## Primary signal

`cone_response` has logical shape `[T, Ncone]`. Each value is a noise-free cone
excitation measured in isomerizations per integration time. The export retains:

- irregular cone positions in retinal degrees;
- cone type identifiers;
- time axis and fixational eye-movement trace;
- optical/display settings and physical units.

The adapter does not rasterize the mosaic into LMS image planes. This keeps the
per-cone representation expected by the architecture and allows local
Bipolar-to-RGC connectivity to use the true cone geometry.

## Run

Set `ISETBIO_ROOT` and `ISETCAM_ROOT`, then run:

```powershell
python external/isetbio_pipeline/run_export.py `
  --input data/isetbio/input_movie.mp4 `
  --output results/isetbio/input_cone_response_movie.h5 `
  --time-steps 8
```

For videos or frame directories, `--time-steps` caps the number of exported
frames. For still images, it keeps the old compatibility behavior and repeats a
fixed-fixation response over the requested time steps. Add `--eye-movements`
only for still-image exports when MATLAB's Statistics and Machine Learning
Toolbox is available.

The command also writes `input_cone_response_preview.png`, a time-mean scatter
plot on the native cone mosaic.

## Normalization boundary

The HDF5 file stores physical cone responses. Per-cone normalization belongs in
the PyTorch input pipeline and must use training-set statistics:

```python
from external.isetbio_pipeline.cone_response_io import (
    apply_per_cone_normalizer,
    fit_per_cone_normalizer,
    load_cone_response,
)

sample = load_cone_response("results/isetbio/input_cone_response.h5")
mean, scale = fit_per_cone_normalizer([sample.response])
x = apply_per_cone_normalizer(sample.response, mean, scale)
```

For real training, fit `mean` and `scale` over all training sequences before
normalizing validation or test data.
