# Retina SNN v9 匿名双通路与动态 RF 执行报告

## 1. 执行状态和一句话结论

状态：COMPLETED。current representation=GO，bank emergence=NO-GO，dynamic RF=NO-GO。

## 2. 研究目标及允许的结论层级

在预设 RGC 输出、ON/OFF 极性和两个匿名对称 bank 下，检验功能身份是否由训练形成，并检验 preceding context 是否通过内部状态改变 RGC-level effective RF。本轮只允许声称单 seed 的机制方向证据。

## 3. 原 v8 问题与本次修改映射

移除了 population-specific support、mix bounds、decoder warmup、decoder spatial logits、无条件二次 energy cost、短 t_bptt 和每轮完整 train_eval。改为对称匿名 bank、tied decoder、256-step 可微窗口和 inequality energy budget。

## 4. 实际修改文件、关键类/函数和 diff 摘要

`models/v9_retina.py`, `training/v9.py`, `evaluation/v9.py`, `scripts/run_v9_emergent_rgc_dynamic_rf.py`, `configs/v9_emergent_rgc_dynamic_rf.yaml`, `tests/test_v9_emergent_rgc.py`

## 5. 数据审计、实际启用的数据模式和限制

{
  "mode": "synthetic_training_noise",
  "train_files": [
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100075_seed301.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100080_seed302.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100098_seed303.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\103041_seed304.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\104022_seed305.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\105019_seed306.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\105053_seed307.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\106020_seed308.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\106025_seed309.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\108041_seed310.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\108073_seed311.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\109034_seed312.h5"
  ],
  "validation_files": [
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\101085_seed401.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\101087_seed402.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\102061_seed403.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\103070_seed404.h5"
  ],
  "train_source_ids": [
    "100075.jpg",
    "100080.jpg",
    "100098.jpg",
    "103041.jpg",
    "104022.jpg",
    "105019.jpg",
    "105053.jpg",
    "106020.jpg",
    "106025.jpg",
    "108041.jpg",
    "108073.jpg",
    "109034.jpg"
  ],
  "validation_source_ids": [
    "101085.jpg",
    "101087.jpg",
    "102061.jpg",
    "103070.jpg"
  ],
  "source_disjoint": true,
  "schema": {
    "response_shape": [
      320,
      29
    ],
    "positions_shape": [
      29,
      2
    ],
    "cone_types_shape": [
      29
    ],
    "time_axis_shape": [
      320
    ],
    "eye_trace_shape": [
      320,
      2
    ],
    "dtype": "float32",
    "dt_ms": 5.000000000000004,
    "eccentricity_deg": 4.0,
    "stimulus_source_kind": "natural_image_microdrift",
    "response_units": "isomerizations_per_integration_time",
    "paired_noisy_clean": false
  },
  "normalization": "per-cone train-only log mean/std; validation reuses train stats"
}

## 6. 最终 resolved config

{
  "seed": 19,
  "train_glob": "data/isetbio_bsds300_4deg/train/*.h5",
  "validation_glob": "data/isetbio_bsds300_4deg/val/*.h5",
  "dt_ms": 5,
  "sequence_steps": 320,
  "burn_in_steps": 64,
  "differentiable_steps": 256,
  "context_only_steps": 160,
  "supervised_steps": 96,
  "checkpoint_block_steps": 32,
  "batch_size": 1,
  "gradient_accumulation_steps": 4,
  "gradient_clip_norm": 1.0,
  "tau_max_ms": 250,
  "noise_std_min": 0.1,
  "noise_std_max": 0.25,
  "context_transition_probability": 0.75,
  "context_gain_min": 0.3,
  "context_gain_max": 1.0,
  "energy_bootstrap_fraction": 0.1,
  "energy_budget_ratio": 0.9,
  "rho_energy": 1.0,
  "dual_lr": 0.01,
  "dual_max": 10.0,
  "lambda_wiring": 0.001,
  "lambda_cross_bank_redundancy": 0.001,
  "lambda_homeostasis": 0.001,
  "lambda_unit_residual": 1e-05,
  "core_and_bank_lr": 0.0002,
  "decoder_scalar_lr": 0.0001,
  "weight_decay": 0.0,
  "lr_warmup_fraction": 0.05,
  "lr_schedule": "cosine",
  "max_optimizer_steps": 300,
  "min_optimizer_steps": 300,
  "validation_interval_steps": 50,
  "early_stopping_patience_validations": 8,
  "dynamic_rf_context_pairs": 32,
  "dynamic_rf_max_units_per_bank_polarity": 8,
  "dynamic_rf_lag_steps": 8,
  "dynamic_rf_recovery_delays_ms": [
    0,
    100,
    200,
    300,
    500
  ],
  "local_linear_baseline": "not_run",
  "device": "cuda",
  "train_files": [
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100075_seed301.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100080_seed302.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\100098_seed303.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\103041_seed304.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\104022_seed305.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\105019_seed306.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\105053_seed307.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\106020_seed308.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\106025_seed309.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\108041_seed310.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\108073_seed311.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\train\\109034_seed312.h5"
  ],
  "validation_files": [
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\101085_seed401.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\101087_seed402.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\102061_seed403.h5",
    "D:\\PythonProject\\retina_rf_SNN\\data\\isetbio_bsds300_4deg\\val\\103070_seed404.h5"
  ]
}

## 7. BPTT 实现和 256-vs-320 梯度审计

{
  "cosine": 1.0,
  "norm_ratio": 1.000006181542525,
  "norm_256": 0.006553830578923225,
  "norm_320": 0.00655379006639123,
  "passed": true
}

## 8. smoke test 与单元测试结果

{
  "command": [
    "D:\\anaconda\\envs\\retina_rf_snn\\python.exe",
    "-m",
    "pytest",
    "-q",
    "tests/test_v9_emergent_rgc.py"
  ],
  "exit_code": 0,
  "output": "................                                                         [100%]\n16 passed in 4.36s",
  "smoke": {
    "optimizer_steps": 2,
    "rows": [
      {
        "loss_total": 0.3998403586447239,
        "raw_reconstruction": 0.32953118719160557,
        "normalized_reconstruction": 0.3988397140055895,
        "energy": 0.25360317900776863,
        "energy_penalty": 0.0,
        "energy_violation": 0.0,
        "wiring": 0.15891312062740326,
        "redundancy": 0.8417311608791351,
        "homeostasis": 0.0,
        "residual": 0.00010051780554931611,
        "bank_a_rate": 0.06326641421765089,
        "bank_b_rate": 0.06326822424307466,
        "energy_budget": 0.0,
        "energy_ema": 0.25360317900776863,
        "dual": 0.0,
        "lr_core": 8.000000000000001e-06,
        "lr_decoder": 4.000000000000001e-06,
        "gradient_norm": 0.46948468685150146,
        "temporal_gradient_norm": 0.01195236761122942
      },
      {
        "loss_total": 0.3063361719250679,
        "raw_reconstruction": 0.2522684168070555,
        "normalized_reconstruction": 0.30532667599618435,
        "energy": 0.24629579111933708,
        "energy_penalty": 0.0,
        "energy_violation": 0.0,
        "wiring": 0.15891222655773163,
        "redundancy": 0.8505906462669373,
        "homeostasis": 0.0,
        "residual": 0.00010051922436105087,
        "bank_a_rate": 0.06331351632252336,
        "bank_b_rate": 0.06321163522079587,
        "energy_budget": 0.0,
        "energy_ema": 0.2532378096133471,
        "dual": 0.0,
        "lr_core": 1.2e-05,
        "lr_decoder": 6e-06,
        "gradient_norm": 0.35157427191734314,
        "temporal_gradient_norm": 0.009110525250434875
      }
    ]
  }
}

## 9. 完整训练命令、环境、GPU、runtime 和 peak memory

{
  "python": "3.11.15 | packaged by conda-forge | (main, Mar  5 2026, 16:36:00) [MSC v.1944 64 bit (AMD64)]",
  "python_executable": "D:\\anaconda\\envs\\snn_env\\python.exe",
  "torch": "2.10.0+cu126",
  "cuda_runtime": "12.6",
  "cuda_available": true,
  "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
  "gpu_total_memory_bytes": 8585216000,
  "command": "D:\\anaconda\\envs\\snn_env\\python.exe scripts/run_v9_emergent_rgc_dynamic_rf.py --config runs/v9_emergent_rgc_dynamic_rf_seed19_20260721_204148/resolved_config.yaml --device cuda --seed 19 --resume-run runs/v9_emergent_rgc_dynamic_rf_seed19_20260721_204148",
  "runtime_seconds": 249.7431938648224,
  "peak_gpu_memory_bytes": 0,
  "optimizer_steps": 301,
  "full_bptt": false
}

## 10. 训练曲线和稳定性

见 `plots/training_curves.png`；NaN/Inf=False。

## 11. checkpoint 选择

主 checkpoint：`D:\PythonProject\retina_rf_SNN\runs\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\checkpoints\best_current.pt`。

## 12. reconstruction 与 energy 结果

{
  "reconstruction": {
    "validation_raw_current_mse": 1.1486085951328278,
    "validation_normalized_current_mse": 1.390189290046692,
    "zero_baseline_mse": 1.2237470149993896,
    "train_fit_global_mean_baseline_mse": 1.2237472534179688,
    "skill_score": 0.06140028857729174,
    "local_linear_baseline": {
      "status": "not_run",
      "reason": "disabled_by_default_to_avoid_high_dimensional_normal_equations",
      "estimated_feature_count_per_target": 5441
    }
  },
  "resources": {
    "energy": 0.4005590006709099,
    "budget": 0.22266166239380508,
    "violation": 0.7989580979705031,
    "dual": 0.036872247699503376,
    "bank_polarity_rates": [
      [
        0.059360407292842865,
        0.1349392980337143
      ],
      [
        0.05888093262910843,
        0.13462266325950623
      ]
    ],
    "active_unit_fraction": [
      [
        0.8448275923728943,
        1.0
      ],
      [
        0.8448275923728943,
        1.0
      ]
    ]
  }
}

## 13. anonymous bank 功能分化结果

{
  "banks": [
    {
      "bank": "A",
      "unit_count": 29,
      "sigma_degs": 0.026135003194212914,
      "effective_radius_degs": 0.03127424791455269,
      "sustained_mix": 0.4992232620716095,
      "transient_mix": 0.5007767677307129,
      "membrane_tau_ms": 19.890472412109375,
      "adaptation_tau_ms": 79.52104187011719,
      "adaptation_gain": 0.09664680808782578,
      "rate_tau_ms": 50.202789306640625,
      "amacrine_gain": 0.01919712871313095,
      "threshold": 0.19907718896865845,
      "mean_rate": 0.09714984893798828,
      "active_fraction": 1.0,
      "decoder_magnitude_on": 0.10018237680196762,
      "decoder_magnitude_off": 0.1001395732164383,
      "unit_radius_degs": [
        0.03375336527824402,
        0.033191125839948654,
        0.030842486768960953,
        0.032665155827999115,
        0.033409010618925095,
        0.03119179606437683,
        0.028922218829393387,
        0.02907753922045231,
        0.030437171459197998,
        0.030530747026205063,
        0.030824363231658936,
        0.028813904151320457,
        0.030733034014701843,
        0.0328807458281517,
        0.031509313732385635,
        0.028595291078090668,
        0.030509864911437035,
        0.0319598913192749,
        0.031829945743083954,
        0.028091352432966232,
        0.030952639877796173,
        0.03143569082021713,
        0.02971433475613594,
        0.03247873857617378,
        0.033091600984334946,
        0.03407901152968407,
        0.03120686300098896,
        0.03337642922997475,
        0.030849523842334747
      ],
      "unit_mean_rate": [
        0.07990654557943344,
        0.10486910492181778,
        0.12104777991771698,
        0.0832492783665657,
        0.09614008665084839,
        0.08633390814065933,
        0.12022469937801361,
        0.1114988774061203,
        0.08711591362953186,
        0.101399727165699,
        0.09075546264648438,
        0.11251170933246613,
        0.10096167027950287,
        0.08597061038017273,
        0.08426834642887115,
        0.09142228960990906,
        0.11971830576658249,
        0.09240179508924484,
        0.0874946266412735,
        0.09461314976215363,
        0.09237425029277802,
        0.08975660055875778,
        0.09559780359268188,
        0.09452954679727554,
        0.09045278280973434,
        0.09245690703392029,
        0.1115947812795639,
        0.10200244188308716,
        0.09667661786079407
      ],
      "impulse_peak_ms": 0.0,
      "impulse_integration_width_ms": 0.0,
      "step_peak_ms": 0.0,
      "step_integration_width_ms": 0.0,
      "flicker_peak_ms": 0.0,
      "flicker_integration_width_ms": 0.0,
      "step_sustained_index": 0.0
    },
    {
      "bank": "B",
      "unit_count": 29,
      "sigma_degs": 0.026293262839317322,
      "effective_radius_degs": 0.03140478953719139,
      "sustained_mix": 0.5009088516235352,
      "transient_mix": 0.49909114837646484,
      "membrane_tau_ms": 20.002805709838867,
      "adaptation_tau_ms": 80.27387237548828,
      "adaptation_gain": 0.10059414058923721,
      "rate_tau_ms": 50.70518112182617,
      "amacrine_gain": 0.020650291815400124,
      "threshold": 0.19908234477043152,
      "mean_rate": 0.09675179421901703,
      "active_fraction": 1.0,
      "decoder_magnitude_on": 0.10018223524093628,
      "decoder_magnitude_off": 0.10013903677463531,
      "unit_radius_degs": [
        0.03399457782506943,
        0.03337275609374046,
        0.031003566458821297,
        0.032816123217344284,
        0.033561673015356064,
        0.03131618723273277,
        0.029044359922409058,
        0.02907063066959381,
        0.03057124838232994,
        0.03068152628839016,
        0.03080209158360958,
        0.028827395290136337,
        0.030880814418196678,
        0.03301446512341499,
        0.031641725450754166,
        0.028678344562649727,
        0.03072292171418667,
        0.03207501024007797,
        0.03196373209357262,
        0.028219955042004585,
        0.031009381636977196,
        0.03159274160861969,
        0.02980682998895645,
        0.03264743089675903,
        0.033185891807079315,
        0.03432226926088333,
        0.031391680240631104,
        0.033508140593767166,
        0.031015394255518913
      ],
      "unit_mean_rate": [
        0.07941615581512451,
        0.10431882739067078,
        0.1211479902267456,
        0.08199169486761093,
        0.09570731222629547,
        0.08599153161048889,
        0.12039132416248322,
        0.1114283874630928,
        0.08646196126937866,
        0.10134781897068024,
        0.09074260294437408,
        0.11247280985116959,
        0.10068650543689728,
        0.08586154133081436,
        0.08300332725048065,
        0.09150058776140213,
        0.11929135024547577,
        0.09241892397403717,
        0.08749628067016602,
        0.09441179037094116,
        0.09137310832738876,
        0.08895374834537506,
        0.09412826597690582,
        0.09391483664512634,
        0.09035750478506088,
        0.09288311749696732,
        0.11045540869235992,
        0.1013215109705925,
        0.0963258445262909
      ],
      "impulse_peak_ms": 0.0,
      "impulse_integration_width_ms": 0.0,
      "step_peak_ms": 0.0,
      "step_integration_width_ms": 0.0,
      "flicker_peak_ms": 0.0,
      "flicker_integration_width_ms": 0.0,
      "step_sustained_index": 0.0
    }
  ],
  "differences": {
    "effective_radius_a_minus_b": -0.0001305416808463633,
    "effective_radius_bootstrap_ci": [
      -0.00015219383894873317,
      -0.00010825310400832677
    ],
    "mean_rate_a_minus_b": 0.00039805343840271235,
    "mean_rate_bootstrap_ci": [
      0.00023519583155575673,
      0.0005558906093938276
    ],
    "step_sustained_index_a_minus_b": 0.0
  },
  "unit_feature_silhouette": 0.35045570135116577,
  "role_assignment": {
    "bank_a": "bank A",
    "bank_b": "bank B",
    "status": "未形成预期功能配对"
  }
}

## 14. dynamic RF matched-context、state reset 和 recovery 结果

{
  "status": "not_identifiable",
  "context_pair_count": 32,
  "representative_units_per_bank_polarity": 1,
  "lag_steps": 8,
  "pair_metrics": [
    {
      "raw_kernel_norm_ratio": 0.37966927886009216,
      "gain_normalized_cosine_distance": 0.2119966745376587,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": 0.07248878479003906,
      "spatial_second_moment_shift": 0.00015112675464479253
    },
    {
      "raw_kernel_norm_ratio": 0.6764897257089615,
      "gain_normalized_cosine_distance": 0.14078117907047272,
      "temporal_peak_shift_ms": 6.250000000000005,
      "temporal_integration_width_shift_ms": -0.19166254997253418,
      "spatial_second_moment_shift": 0.0001224968582391739
    },
    {
      "raw_kernel_norm_ratio": 0.4749235659837723,
      "gain_normalized_cosine_distance": 0.10872334241867065,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": 0.8201615810394287,
      "spatial_second_moment_shift": 0.00011653936235234141
    },
    {
      "raw_kernel_norm_ratio": 0.6539590805768967,
      "gain_normalized_cosine_distance": 0.029251739382743835,
      "temporal_peak_shift_ms": 3.7500000000000036,
      "temporal_integration_width_shift_ms": 0.4679105281829834,
      "spatial_second_moment_shift": 7.368461228907108e-05
    },
    {
      "raw_kernel_norm_ratio": 0.2978728376328945,
      "gain_normalized_cosine_distance": 0.15731500089168549,
      "temporal_peak_shift_ms": -1.250000000000001,
      "temporal_integration_width_shift_ms": 0.470806360244751,
      "spatial_second_moment_shift": 0.0003316113925393438
    },
    {
      "raw_kernel_norm_ratio": 0.628004252910614,
      "gain_normalized_cosine_distance": 0.1191660463809967,
      "temporal_peak_shift_ms": 1.2500000000000009,
      "temporal_integration_width_shift_ms": -0.32830822467803955,
      "spatial_second_moment_shift": -9.319726086687297e-05
    },
    {
      "raw_kernel_norm_ratio": 0.6698251888155937,
      "gain_normalized_cosine_distance": 0.0564231276512146,
      "temporal_peak_shift_ms": 2.5000000000000018,
      "temporal_integration_width_shift_ms": 0.471341609954834,
      "spatial_second_moment_shift": 5.7604855101089925e-05
    },
    {
      "raw_kernel_norm_ratio": 0.6731208264827728,
      "gain_normalized_cosine_distance": 0.12715716660022736,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": 0.21919012069702148,
      "spatial_second_moment_shift": -3.350083716213703e-06
    },
    {
      "raw_kernel_norm_ratio": 0.7700174376368523,
      "gain_normalized_cosine_distance": 0.13767598569393158,
      "temporal_peak_shift_ms": 2.5000000000000018,
      "temporal_integration_width_shift_ms": -0.5057723522186279,
      "spatial_second_moment_shift": 4.695955431088805e-05
    },
    {
      "raw_kernel_norm_ratio": 0.34430060535669327,
      "gain_normalized_cosine_distance": 0.1250307410955429,
      "temporal_peak_shift_ms": 5.0000000000000036,
      "temporal_integration_width_shift_ms": -0.7769591808319092,
      "spatial_second_moment_shift": 0.00013665468577528372
    },
    {
      "raw_kernel_norm_ratio": 0.5821557939052582,
      "gain_normalized_cosine_distance": 0.053950220346450806,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": -0.5957546234130859,
      "spatial_second_moment_shift": 6.198214396135882e-05
    },
    {
      "raw_kernel_norm_ratio": 0.4890259578824043,
      "gain_normalized_cosine_distance": 0.05944165587425232,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": 0.2688026428222656,
      "spatial_second_moment_shift": 3.597597242332995e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5452234968543053,
      "gain_normalized_cosine_distance": 0.07368579506874084,
      "temporal_peak_shift_ms": 10.000000000000009,
      "temporal_integration_width_shift_ms": -0.3363626003265381,
      "spatial_second_moment_shift": 5.239488382358104e-05
    },
    {
      "raw_kernel_norm_ratio": 0.37216177210211754,
      "gain_normalized_cosine_distance": 0.16918013989925385,
      "temporal_peak_shift_ms": -2.500000000000002,
      "temporal_integration_width_shift_ms": -0.45711517333984375,
      "spatial_second_moment_shift": 0.00036635177093558013
    },
    {
      "raw_kernel_norm_ratio": 0.5453731119632721,
      "gain_normalized_cosine_distance": 0.0541645884513855,
      "temporal_peak_shift_ms": 2.500000000000002,
      "temporal_integration_width_shift_ms": -0.9576930999755859,
      "spatial_second_moment_shift": 8.798004273558035e-05
    },
    {
      "raw_kernel_norm_ratio": 0.46893392875790596,
      "gain_normalized_cosine_distance": 0.0995045155286789,
      "temporal_peak_shift_ms": 2.500000000000002,
      "temporal_integration_width_shift_ms": 0.8858497142791748,
      "spatial_second_moment_shift": 7.888118852861226e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5036219134926796,
      "gain_normalized_cosine_distance": 0.1968417465686798,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": -0.20395874977111816,
      "spatial_second_moment_shift": 0.0003122884809272364
    },
    {
      "raw_kernel_norm_ratio": 0.6933364272117615,
      "gain_normalized_cosine_distance": 0.11073468625545502,
      "temporal_peak_shift_ms": 2.5000000000000027,
      "temporal_integration_width_shift_ms": 0.05780625343322754,
      "spatial_second_moment_shift": 0.00013757075066678226
    },
    {
      "raw_kernel_norm_ratio": 0.4206974692642689,
      "gain_normalized_cosine_distance": 0.026524528861045837,
      "temporal_peak_shift_ms": -5.000000000000004,
      "temporal_integration_width_shift_ms": -0.2167830467224121,
      "spatial_second_moment_shift": 0.0001321079907938838
    },
    {
      "raw_kernel_norm_ratio": 0.8425834625959396,
      "gain_normalized_cosine_distance": 0.0866771936416626,
      "temporal_peak_shift_ms": 3.7500000000000027,
      "temporal_integration_width_shift_ms": -0.12607073783874512,
      "spatial_second_moment_shift": 2.991007931996137e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5073790326714516,
      "gain_normalized_cosine_distance": 0.143675297498703,
      "temporal_peak_shift_ms": 2.500000000000002,
      "temporal_integration_width_shift_ms": 0.6778807640075684,
      "spatial_second_moment_shift": 0.00021523483155760914
    },
    {
      "raw_kernel_norm_ratio": 0.6636583842337132,
      "gain_normalized_cosine_distance": 0.08047035336494446,
      "temporal_peak_shift_ms": 8.750000000000007,
      "temporal_integration_width_shift_ms": -0.07514238357543945,
      "spatial_second_moment_shift": 9.98879040707834e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5883840136229992,
      "gain_normalized_cosine_distance": 0.031310200691223145,
      "temporal_peak_shift_ms": -2.5000000000000027,
      "temporal_integration_width_shift_ms": -0.09492194652557373,
      "spatial_second_moment_shift": 6.94612244842574e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5704491101205349,
      "gain_normalized_cosine_distance": 0.12029266357421875,
      "temporal_peak_shift_ms": 1.2500000000000009,
      "temporal_integration_width_shift_ms": -0.09835362434387207,
      "spatial_second_moment_shift": 4.03400044888258e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5217119827866554,
      "gain_normalized_cosine_distance": 0.19996856153011322,
      "temporal_peak_shift_ms": 5.0000000000000036,
      "temporal_integration_width_shift_ms": -0.46352672576904297,
      "spatial_second_moment_shift": 0.00019670631445478648
    },
    {
      "raw_kernel_norm_ratio": 0.41607868671417236,
      "gain_normalized_cosine_distance": 0.026624545454978943,
      "temporal_peak_shift_ms": -2.500000000000002,
      "temporal_integration_width_shift_ms": 0.14767217636108398,
      "spatial_second_moment_shift": 0.0001405397051712498
    },
    {
      "raw_kernel_norm_ratio": 0.6226776093244553,
      "gain_normalized_cosine_distance": 0.023292362689971924,
      "temporal_peak_shift_ms": -12.50000000000001,
      "temporal_integration_width_shift_ms": 0.05968952178955078,
      "spatial_second_moment_shift": 8.724354847799987e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5055969506502151,
      "gain_normalized_cosine_distance": 0.15177178382873535,
      "temporal_peak_shift_ms": 7.500000000000007,
      "temporal_integration_width_shift_ms": -0.7741312980651855,
      "spatial_second_moment_shift": 0.0003872571396641433
    },
    {
      "raw_kernel_norm_ratio": 0.48983200639486313,
      "gain_normalized_cosine_distance": 0.14866873621940613,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": -0.5471369028091431,
      "spatial_second_moment_shift": 6.156596646178514e-05
    },
    {
      "raw_kernel_norm_ratio": 0.5464419201016426,
      "gain_normalized_cosine_distance": 0.015551969408988953,
      "temporal_peak_shift_ms": 0.0,
      "temporal_integration_width_shift_ms": 0.005383968353271484,
      "spatial_second_moment_shift": 0.00014782030484639108
    },
    {
      "raw_kernel_norm_ratio": 0.629771813750267,
      "gain_normalized_cosine_distance": 0.048116058111190796,
      "temporal_peak_shift_ms": 2.500000000000002,
      "temporal_integration_width_shift_ms": 0.17563295364379883,
      "spatial_second_moment_shift": 0.00012694986071437597
    },
    {
      "raw_kernel_norm_ratio": 0.9773487597703934,
      "gain_normalized_cosine_distance": 0.022043630480766296,
      "temporal_peak_shift_ms": -8.881784197001252e-16,
      "temporal_integration_width_shift_ms": 0.16031980514526367,
      "spatial_second_moment_shift": -7.334718247875571e-06
    }
  ],
  "summary": {
    "raw_kernel_norm_ratio": 0.5647070751292631,
    "raw_kernel_norm_ratio_bootstrap_ci": [
      0.5189185543596977,
      0.6138592994393548
    ],
    "gain_normalized_cosine_distance": 0.09862538240849972,
    "gain_normalized_cosine_distance_bootstrap_ci": [
      0.07863343970384448,
      0.11769751934334635
    ],
    "temporal_peak_shift_ms": 1.367187500000001,
    "temporal_peak_shift_ms_bootstrap_ci": [
      -0.07812500000000011,
      2.734375000000002
    ],
    "temporal_integration_width_shift_ms": -0.055897388607263565,
    "temporal_integration_width_shift_ms_bootstrap_ci": [
      -0.20678030624985694,
      0.11230933386832474
    ],
    "spatial_second_moment_shift": 0.0001187889412790355,
    "spatial_second_moment_shift_bootstrap_ci": [
      8.31529680496601e-05,
      0.00015673616619409357
    ]
  },
  "state_reset_control_effect": 0.0,
  "recovery_curve": [
    {
      "delay_ms": 0.0,
      "effect": 0.08786247577518225
    },
    {
      "delay_ms": 100.0,
      "effect": 0.021231914637610316
    },
    {
      "delay_ms": 200.0,
      "effect": 0.0032069830340333283
    },
    {
      "delay_ms": 300.0,
      "effect": 0.0010859781905310228
    },
    {
      "delay_ms": 500.0,
      "effect": 0.00010152039754984798
    }
  ],
  "finite_difference_direction_check": {
    "status": "mismatch",
    "autodiff_directional_derivative": -0.002544572576880455,
    "finite_difference_directional_derivative": 0.0,
    "relative_error": 0.9999960700823425,
    "unit_count": 1,
    "local_cone_count": 8
  }
}

## 15. 参数边界与失败诊断

{
  "parameters": [
    {
      "name": "mix",
      "bank": "A",
      "value": 0.4992232620716095,
      "lower": 0.0,
      "upper": 1.0,
      "fraction": 0.4992232620716095,
      "near_boundary": false
    },
    {
      "name": "mix",
      "bank": "B",
      "value": 0.5009088516235352,
      "lower": 0.0,
      "upper": 1.0,
      "fraction": 0.5009088516235352,
      "near_boundary": false
    },
    {
      "name": "membrane_tau_ms",
      "bank": "A",
      "value": 19.890472412109375,
      "lower": 5.0,
      "upper": 80.0,
      "fraction": 0.19853963216145834,
      "near_boundary": false
    },
    {
      "name": "membrane_tau_ms",
      "bank": "B",
      "value": 20.002805709838867,
      "lower": 5.0,
      "upper": 80.0,
      "fraction": 0.20003740946451823,
      "near_boundary": false
    },
    {
      "name": "adaptation_tau_ms",
      "bank": "A",
      "value": 79.52104187011719,
      "lower": 20.0,
      "upper": 250.0,
      "fraction": 0.2587871385657269,
      "near_boundary": false
    },
    {
      "name": "adaptation_tau_ms",
      "bank": "B",
      "value": 80.27387237548828,
      "lower": 20.0,
      "upper": 250.0,
      "fraction": 0.262060314676036,
      "near_boundary": false
    },
    {
      "name": "adaptation_gain",
      "bank": "A",
      "value": 0.09664680808782578,
      "lower": 0.0,
      "upper": 1.0,
      "fraction": 0.09664680808782578,
      "near_boundary": false
    },
    {
      "name": "adaptation_gain",
      "bank": "B",
      "value": 0.10059414058923721,
      "lower": 0.0,
      "upper": 1.0,
      "fraction": 0.10059414058923721,
      "near_boundary": false
    },
    {
      "name": "rate_tau_ms",
      "bank": "A",
      "value": 50.202789306640625,
      "lower": 5.0,
      "upper": 250.0,
      "fraction": 0.18450118084343112,
      "near_boundary": false
    },
    {
      "name": "rate_tau_ms",
      "bank": "B",
      "value": 50.70518112182617,
      "lower": 5.0,
      "upper": 250.0,
      "fraction": 0.18655175968092316,
      "near_boundary": false
    },
    {
      "name": "amacrine_gain",
      "bank": "A",
      "value": 0.01919712871313095,
      "lower": 0.0,
      "upper": 0.3,
      "fraction": 0.06399042904376984,
      "near_boundary": false
    },
    {
      "name": "amacrine_gain",
      "bank": "B",
      "value": 0.020650291815400124,
      "lower": 0.0,
      "upper": 0.3,
      "fraction": 0.06883430605133375,
      "near_boundary": false
    },
    {
      "name": "threshold",
      "bank": "A",
      "value": 0.19907718896865845,
      "lower": 0.05,
      "upper": 1.0,
      "fraction": 0.15692335680911418,
      "near_boundary": false
    },
    {
      "name": "threshold",
      "bank": "B",
      "value": 0.19908234477043152,
      "lower": 0.05,
      "upper": 1.0,
      "fraction": 0.1569287839688753,
      "near_boundary": false
    },
    {
      "name": "subunit_tau_ms",
      "bank": "A",
      "value": 50.10482406616211,
      "lower": 10.0,
      "upper": 200.0,
      "fraction": 0.21107802140085322,
      "near_boundary": false
    },
    {
      "name": "subunit_tau_ms",
      "bank": "B",
      "value": 49.24555587768555,
      "lower": 10.0,
      "upper": 200.0,
      "fraction": 0.20655555725097657,
      "near_boundary": false
    },
    {
      "name": "subunit_gain",
      "bank": "A",
      "value": 0.5044578909873962,
      "lower": 0.0,
      "upper": 3.0,
      "fraction": 0.16815263032913208,
      "near_boundary": false
    },
    {
      "name": "subunit_gain",
      "bank": "B",
      "value": 0.497437059879303,
      "lower": 0.0,
      "upper": 3.0,
      "fraction": 0.16581235329310098,
      "near_boundary": false
    }
  ],
  "near_boundary_count": 0
}

## 16. 已知事实、合理推断、仍需验证内容

已知事实来自落盘指标；bank 身份和 dynamic RF 解释仅按预注册判据生成。单 seed、12/4 个 source、synthetic noise 和 4° 偏心度限制外推。

## 17. GO / PARTIAL / NO-GO

- current representation：GO
- midget-like/parasol-like 功能配对：NO-GO
- state-dependent dynamic RF：NO-GO（原始状态 `not_identifiable`）

## 18. 下一步（最多三项）

1. 仅在本轮机制方向为 GO/PARTIAL 时增加独立 seed。
2. 用更大 source-disjoint 自然序列复核不确定性。
3. finite-difference 不一致时改进可辨识 readout，而不是增加细胞机制。

## 19. 所有产物的绝对路径

{
  "resolved_config.yaml": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\resolved_config.yaml",
  "environment_manifest.json": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\environment_manifest.json",
  "data_manifest.json": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\data_manifest.json",
  "git_status_diff_summary.txt": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\git_status_diff_summary.txt",
  "training_log.csv": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\training_log.csv",
  "training_log.jsonl": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\training_log.jsonl",
  "validation_log.jsonl": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\validation_log.jsonl",
  "stdout_stderr.log": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\stdout_stderr.log",
  "final_metrics.json": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\final_metrics.json",
  "evaluation_metrics.csv": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\evaluation_metrics.csv",
  "final_report_zh.md": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\final_report_zh.md",
  "run_manifest.json": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\run_manifest.json",
  "run_status.json": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\run_status.json",
  "checkpoint": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\checkpoints\\best_current.pt",
  "training_curves.png": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\plots\\training_curves.png",
  "bank_feature_summary.png": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\plots\\bank_feature_summary.png",
  "dynamic_rf_context_summary.png": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\plots\\dynamic_rf_context_summary.png",
  "dynamic_rf_recovery.png": "D:\\PythonProject\\retina_rf_SNN\\runs\\v9_emergent_rgc_dynamic_rf_seed19_20260721_204148\\plots\\dynamic_rf_recovery.png",
  "docs_report": "D:\\PythonProject\\retina_rf_SNN\\docs\\v9_emergent_rgc_dynamic_rf_execution_report_20260721_235823.md"
}
