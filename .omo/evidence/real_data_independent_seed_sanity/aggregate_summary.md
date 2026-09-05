# 81-combination full-precision aggregate check

STATUS: COMPLETED

18 cells are fixed at saved primary logits. The selected cells are `69#4`, `67#6`, `68#4`, and `67#4`; P/S1/S2 form all 3^4 = 81 assignments.
Each paired logit is the existing 300–400 ms paired trace mean. Inputs are raw saved float32 logits, converted to float64 before the within-cell mean and equal-cell `math.fsum / 22` aggregate. No numeric threshold is applied: reversed means `normal * AC-off < 0`.

| Signature | Reversed combinations | Normal range | AC-off range | Closest normal-to-zero | Closest AC-off-to-zero |
|---|---:|---|---|---|---|
| Mach dark | 81 | [-0.0037947681817141446, -0.0035342836018764615] | [0.0025317188465234005, 0.0027921342488491175] | C49 = -0.0035342836018764615 | C38 = 0.0025317188465234005 |
| Mach bright | 81 | [0.0035342864917986324, 0.0037947708910161799] | [-0.0027921306364464037, -0.0025317155953609584] | C49 = 0.0035342864917986324 | C38 = -0.0025317155953609584 |
| SBC | 81 | [-0.040082018483768807, -0.037265473062341861] | [0.0025162162202777283, 0.0031021490241542006] | C38 = -0.037265473062341861 | C69 = 0.0025162162202777283 |
| Hermann | 81 | [-0.020495447967991683, -0.019042214119073118] | [0.00086875016039068039, 0.0010689146590955329] | C38 = -0.019042214119073118 | C69 = 0.00086875016039068039 |
| White | 81 | [0.018630986683296435, 0.020035130508018265] | [-0.0015652815500895182, -0.0012726007085857968] | C38 = 0.018630986683296435 | C69 = -0.0012726007085857968 |

Combination labels in CSV are in Cartesian-product order `69#4, 67#6, 68#4, 67#4`, each taking `primary`, `fresh_1`, `fresh_2`. `C01` is all-primary.

## Validation

- 81 rows were emitted exactly once.
- The C01 aggregate equals the direct aggregate of all 22 saved primary tensors for every signature and condition.
- For every selected cell, its primary normal and AC-off tensor is bitwise identical to the historical 22-cell frozen application tensor.
- All source tensor hashes used by this aggregate are recorded below.

```json
{
  "D:\\PythonProject\\retina_rf_SNN\\output\\real_data\\schottdorf_canonical_v1_shared_bc_frozen_applications_20260830\\illusion\\inputs.pt": "60605a8c11eccbc7a0fd17f3c866d6a64a4c1a89e6bf1a39c1effb22f41513ab",
  "D:\\PythonProject\\retina_rf_SNN\\output\\real_data\\schottdorf_canonical_v1_shared_bc_frozen_applications_20260830\\illusion\\responses.pt": "9f0b0c4b8612e6aca8e337acb1aad3372db9efef96f939489f90ec20b743bf82",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\69_4\\primary\\evaluation.pt": "5be9418ae8e501a090d01f4468672490664fddcca3dded8ad133a44460d729c1",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\69_4\\fresh_1\\evaluation.pt": "7d1a7f5c0bd2b974b16061738a402cd25979c8b46a9bdc2ed0d1262751575aca",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\69_4\\fresh_2\\evaluation.pt": "8afb1672de786ff2f607983869bd6dcb5b2dcca99fd8feceb1358d4042069130",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_6\\primary\\evaluation.pt": "147151a2caefe5cef834c55980fffdf92e00c701f04a0a954617718ec583c18c",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_6\\fresh_1\\evaluation.pt": "6948ea2c94d11fdb15b14fae6b596176402272bdd8559eeafff999de0fa65519",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_6\\fresh_2\\evaluation.pt": "ed8069afa96b3d2b51efadcd84cd7587285d0e2e8c1376bd36b0a9ac3d8c65cf",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\68_4\\primary\\evaluation.pt": "97c46b70fcc334ef50a75c2870919cada7a8bd7cc0965db59c329bed077e6b98",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\68_4\\fresh_1\\evaluation.pt": "22aad2441ddb3504c377f929cccc60977263c7db29e285908d9aedf08f132db9",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\68_4\\fresh_2\\evaluation.pt": "8809b5125960f19fb85d393a6be34188ae3dbc1f37cb0cec282c63204d98f7b8",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_4\\primary\\evaluation.pt": "2eae8c9e833f096d5479e3249e33b7967d4fe692fd169b896145f7a01da9774b",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_4\\fresh_1\\evaluation.pt": "fba40d1def9dc406dd265e385724db6836d31c286e344030fbfd924c7e616498",
  "D:\\PythonProject\\retina_rf_SNN\\.omo\\evidence\\real_data_independent_seed_sanity\\fits\\67_4\\fresh_2\\evaluation.pt": "11d28b027b1e3ae69cb8e321d3a9ea8b2a9624a9fb478b6e83bccf39e072e30d"
}
```
