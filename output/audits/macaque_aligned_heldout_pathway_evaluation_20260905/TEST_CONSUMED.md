# Test consumption record

This held-out range has now been consumed for confirmatory evaluation.

First successful target-based NLL UTC: 2026-09-05T10:23:17.574463+00:00; first cell/model: 67#4 / zero.

The entire declared [20,60) s range (live frames [3000,9000), decoded movie frames [3751,9751)) is conservatively retired at this first result. Completion at creation: PARTIAL; evaluation_status.json records subsequent completion or failure.

Branch: rgc-readout-v2; HEAD: fea28de038821fadee279b93728688b34bcb3bac; protocol SHA256: 3652c48aa4293f2c3503e15099b40657f6d3b396f4a06f4434c4f0e27167bb08.

Cells: 67#4, 67#6, 67#7, 67#14, 67#21, 67#26, 67#33, 67#34, 68#3, 68#4, 68#7, 68#10, 68#11, 69#3, 69#4, 69#6, 69#7, 69#21, 70#1, 70#7, 70#15, 70#34.

Recordings: lSS01071, lSS01078, lSS01079, lSS01086, lSS01087, lSS01110, lSS01112, lSS01130, lSS01131, lSS01141, lSS01142, lSS01159, lSS01160, lSS01167, lSS01168, lSS01181, lSS01183, lSS01184, lSS01194, lSS01196, lSS01221, lSS01225, lSS01227, lSS01251, lSS01252, lSS01254, lSS01256, lSS01257, lSS01258, lSS01259, lSS01270, lSS01278, lSS01284, lSS01285, lSS01287, lSS01299, lSS01300.

Exact local trials are the immutable TEST_PROTOCOL.md table (137 total). Models: frozen zero Canonical, aligned Canonical, final LN, final CNN; analytical test-target Constant. Aligned conditions: normal, H1-off, direct-BC-off, AC-off; bias/history/remaining parameters fixed. Metrics: valid-bin Bernoulli NLL; paired model/pathway differences and cell-bootstrap intervals; declared logit effects and correlations.

After result-informed changes to architecture, alignment, loss, front-end or center estimation, this range cannot be described as an untouched confirmatory test. No new model selection or range selection is authorized.

Completion update UTC: 2026-09-05T10:25:00.145424+00:00. All 22 cells and all declared frozen conditions completed; source ranges and identities are in test_source_ranges.csv and test_identity_checks.json.
