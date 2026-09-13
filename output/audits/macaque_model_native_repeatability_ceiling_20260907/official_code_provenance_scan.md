# Official code provenance scan

Artifact class: NEW_DETERMINISTIC_DERIVED_ARTIFACT.

Verdict: OFFICIAL_IMPLEMENTATION_NOT_RECOVERED.

One bounded local scan covered all 116 tracked files at cffefb08c760f04c9a951da46061b361d2288e9b, including all 15 notebooks and their checkpoint copies, every code cell, archive XML contents, and binary strings. Search expressions and per-file SHA256 are retained in official_scan_evidence.json. The local repository is shallow: only its available commit can be inspected. Full history and deleted files outside that history are UNVERIFIED; no fetch or alteration of the official checkout was performed.

retinatools/library.py:322-332 compare_to_file reads already processed activity and computes a model-to-data correlation. It does not parse or pair raw repeated spikes. SimpleLowpass is used by model-response routines. Piecharts.ipynb code cells 2, 5, 8, 11 use externally supplied rounded reliability constants. Notebook checkpoint copies contain no recovered raw-repeat reliability producer. No exact routine meeting the STOP condition was recovered. This is a bounded provenance finding, not a claim that such code never existed.

Proceed only with the separately named MODEL_NATIVE_REPEATABILITY_CEILING. Phase R1 remains UNVERIFIED / UNRESOLVED.
