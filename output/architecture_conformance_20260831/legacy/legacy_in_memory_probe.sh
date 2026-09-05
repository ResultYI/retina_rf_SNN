PYTHONDONTWRITEBYTECODE=1 /d/anaconda/python.exe -B - <<'PY'
import json
from dataclasses import asdict
import torch
from evaluation.mechanistic_retina.karamanlis_v1_rf_validation import validate_v1_checkpoint
from models.mechanistic_retina.contracts import ArchitectureMode, MechanisticRetinaConfig
from models.mechanistic_retina.model import build_mechanistic_retina

torch.set_num_threads(1)
torch.manual_seed(319)
minimal_payload = {
    'schema': 'karamanlis_2024_marmoset_rf_geometry_canonical_retina_v1',
    'stage': 'best_trained',
    'model_config': {
        'cell_specific_gains': True,
        'cell_specific_pathway_mixture': False,
        'architecture_mode': 'legacy',
    },
}
validate_v1_checkpoint(minimal_payload)
config = MechanisticRetinaConfig(architecture_mode=ArchitectureMode.LEGACY, cell_specific_gains=True)
axis = torch.linspace(-0.24, 0.24, 7, dtype=torch.float64)
cone_positions = torch.cartesian_prod(axis, axis)
cell_positions = torch.tensor([[0.0, 0.0], [0.06, 0.0]], dtype=torch.float64)
cell_types = ('midget', 'midget')
polarities = ('ON', 'ON')
source = build_mechanistic_retina(config, cone_positions, cell_positions, cell_types, polarities).eval()
config_payload = asdict(config)
config_payload['architecture_mode'] = config.architecture_mode.value
payload = {
    'schema': minimal_payload['schema'],
    'stage': minimal_payload['stage'],
    'model_config': config_payload,
    'model': {name: value.detach().clone() for name, value in source.state_dict().items()},
}
validate_v1_checkpoint(payload)
model_config_values = dict(payload['model_config'])
model_config_values['architecture_mode'] = ArchitectureMode(model_config_values['architecture_mode'])
loaded_config = MechanisticRetinaConfig(**model_config_values)
loaded = build_mechanistic_retina(loaded_config, cone_positions, cell_positions, cell_types, polarities)
load_result = loaded.load_state_dict(payload['model'], strict=True)
loaded.eval()
index = torch.arange(1 * 32 * 49, dtype=torch.float64).reshape(1, 32, 49)
cones = 0.7 * torch.sin(index * 0.173) + 0.2 * torch.cos(index * 0.041)
history = (torch.arange(64).reshape(1, 32, 2) % 7 == 0).to(torch.float64)
with torch.no_grad():
    result = loaded.forward_sequence(cones, observed_counts=history)
    source_result = source.forward_sequence(cones, observed_counts=history)
print(json.dumps({
    'fixture': 'in_memory_only_no_checkpoint_io',
    'seed': 319,
    'torch': torch.__version__,
    'dtype': str(cones.dtype),
    'input_shape': list(cones.shape),
    'minimal_validator_accepts_legacy': True,
    'full_model_payload_validator_accepts_legacy': True,
    'loaded_architecture_mode': loaded.config.architecture_mode.value,
    'strict_load_missing_keys': load_result.missing_keys,
    'strict_load_unexpected_keys': load_result.unexpected_keys,
    'state_dict_key_count': len(payload['model']),
    'root_load_pre_hook_count': len(loaded._load_state_dict_pre_hooks),
    'logits_shape': list(result.logits.shape),
    'logits_all_finite': bool(torch.isfinite(result.logits).all()),
    'reloaded_vs_source_logits_max_abs_difference': float((result.logits-source_result.logits).abs().max()),
    'direct_broad_path_spatial_basis_equal': torch.equal(loaded.feature_bank.path_spatial_basis[:, :2], loaded.feature_bank.path_spatial_basis[:, 2:]),
    'direct_broad_output_equal': torch.equal(result.bc_direct_presynaptic, result.bc_broad_presynaptic),
    'direct_broad_output_max_abs_difference': float((result.bc_direct_presynaptic-result.bc_broad_presynaptic).abs().max()),
    'bc_ac_support_buffers_equal': torch.equal(loaded.feature_bank.bc_support, loaded.feature_bank.ac_support),
    'trainable_parameter_keys': [name for name, parameter in loaded.named_parameters() if parameter.requires_grad],
}, indent=2))
PY
