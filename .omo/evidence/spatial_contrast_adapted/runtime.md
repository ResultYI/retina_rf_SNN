# Runtime preparation

- The existing base interpreter is `D:/anaconda/python.exe`, Python 3.12.7, PyTorch 2.6.0+cpu, NumPy 2.2.6, SciPy 1.13.1.
- The first test could not collect: importing `scipy.optimize` failed with `numpy.core.multiarray failed to import`; SciPy's compiled module reported NumPy 1.x/2.x ABI incompatibility.
- Hypothesis: new baseline logic caused the failure — rejected, no implementation existed yet and the traceback fails inside SciPy import.
- Hypothesis: a project module shadows NumPy/SciPy — rejected, imported paths are the base interpreter's site-packages.
- Hypothesis: the base SciPy binary is incompatible with NumPy 2.2.6 — directly supported by the exception. The separate snn_env imports SciPy 1.17.1 successfully, but uses Python 3.11/PyTorch 2.10 and lacks cv2; it cannot preserve the original data/LN runtime.
- Scoped action: install only SciPy 1.17.1 into `.tmp/spatial-contrast-scipy`, using the repository uv cache. Use that package path only for this process. Preserve base NumPy, PyTorch, cv2, all global packages and all model/data source files. No optimization settings are changed.
- No debugger, listener, instrumentation, model fit or persistent environment change was created during this environment check. Runtime evidence is retained rather than deleted.
