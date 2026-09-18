from __future__ import annotations

import torch
from run import (ROOT, OUT, CONDITIONS, SEEDS, INTERVENTIONS, build, checked_protocol, read_json,
    sha, state_sha, h_observation, b_observation, write_json, configure_runtime, utc,
    observe_mechanism, Intervention, InterventionSpec)
from evaluation.mechanistic_retina.clean_sampled_data import _sample_spikes


def verify():
    p = checked_protocol()
    meta = read_json(OUT / "teacher_metadata.json")
    cp = torch.load(ROOT / p["teacher"]["path"], weights_only=True)
    teacher = build(meta, cp["model"]).eval().requires_grad_(False)
    before = state_sha(teacher.state_dict())
    node = p["observation"]["h1_node_index"]
    r = torch.load(OUT / "datasets/D_R_train.pt", weights_only=True)
    x, y = r["x"][:2], r["target"][:2]
    checks = {}
    with torch.no_grad():
        original = teacher(x, observed_counts=y)
        normal = observe_mechanism(teacher, x, observed_counts=y)
        assert torch.equal(original.logits, normal.logits)
        assert torch.equal(b_observation(teacher, x), torch.cat((original.bc_direct_presynaptic, original.bc_broad_presynaptic), -1).squeeze(2))
        assert torch.equal(h_observation(teacher, x, node), original.h1_state[..., node:node+1])
        checks["identity_observation_matches_official_forward"] = True
        changed = y.clone(); changed[:, 70:] = 1 - changed[:, 70:]
        future = teacher(x, observed_counts=changed).logits
        assert torch.equal(original.logits[:, :71], future[:, :71])
        changed_trace = observe_mechanism(teacher, x, observed_counts=changed)
        assert torch.equal(normal.tensors["history_state"][:, :71], changed_trace.tensors["history_state"][:, :71])
        assert not torch.equal(normal.tensors["history_state"][:, 71:], changed_trace.tensors["history_state"][:, 71:])
        altered_x = x.clone(); altered_x[:, 71:] = -altered_x[:, 71:]
        torch.testing.assert_close(teacher(altered_x, observed_counts=y).logits[:, :71], original.logits[:, :71], rtol=0, atol=0)
        checks["no_current_or_future_target_history_and_no_future_stimulus"] = True
        single = teacher(x[:1], observed_counts=y[:1]).logits
        torch.testing.assert_close(single, original.logits[:1], rtol=1e-5, atol=2e-6)
        checks["independent_sequence_reset"] = True
        seed = 91827
        sampled = _sample_spikes(teacher, x, trials=1, seed=seed)[:, 0]
        true_p = teacher(x, observed_counts=sampled).spike_probability
        rng = torch.Generator().manual_seed(seed)
        expected = torch.stack([(torch.rand((2, 1), generator=rng) < true_p[:, t]).float() for t in range(150)], 1)
        assert torch.equal(sampled, expected)
        for t in (0, 1, 30, 70, 149):
            prefix = teacher(x[:, :t+1], observed_counts=sampled[:, :t+1]).spike_probability[:, -1]
            torch.testing.assert_close(prefix, true_p[:, t], rtol=1e-5, atol=2e-6)
        checks["causal_sampler_replay_and_prefix_check"] = True
        for family, operation in (("H", lambda z: h_observation(teacher, z, node)), ("B", lambda z: b_observation(teacher, z))):
            data = torch.load(OUT / f"datasets/D_{family}_train.pt", weights_only=True)
            torch.testing.assert_close(operation(data["x"][:2]), data["target"][:2], rtol=1e-5, atol=2e-6)
            v = data["target"][:, 30:].double().reshape(-1, data["target"].shape[-1])
            norm = read_json(OUT / "normalization.json")[family]
            torch.testing.assert_close(v.mean(0), torch.tensor(norm["mean"], dtype=torch.float64), rtol=0, atol=0)
            torch.testing.assert_close(v.std(0, correction=0), torch.tensor(norm["std"], dtype=torch.float64), rtol=0, atol=0)
        checks["teacher_train_only_normalization_and_targets"] = True
        zeros = torch.zeros_like(y)
        base = observe_mechanism(teacher, x, observed_counts=zeros, intervention=InterventionSpec(Intervention.FIX_HISTORY_ZERO))
        for name in INTERVENTIONS:
            trace = observe_mechanism(teacher, x, observed_counts=zeros, intervention=InterventionSpec(Intervention(name)))
            assert not torch.count_nonzero(trace.tensors["history_state"])
            assert not torch.count_nonzero(trace.tensors["history_term"])
            assert torch.isfinite(base.logits - trace.logits).all()
        checks["formal_interventions_zero_history"] = True
    initial_hashes = []
    for seed in SEEDS:
        initial = torch.load(OUT / "initial_states" / f"{seed}.pt", weights_only=True)
        model = build(meta, initial["model"])
        initial_hashes.append(state_sha(model.state_dict()))
        assert initial_hashes[-1] != before
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 33
    assert len(set(initial_hashes)) == 3
    checks["three_distinct_standard_initializations_no_teacher_warm_start"] = True
    model = build(meta, initial["model"])
    h = torch.load(OUT / "datasets/D_H_train.pt", weights_only=True)
    b = torch.load(OUT / "datasets/D_B_train.pt", weights_only=True)
    norm = read_json(OUT / "normalization.json")
    def losses():
        lh = ((h_observation(model, h["x"][:2], node)[:, 30:] - h["target"][:2, 30:]) / torch.tensor(norm["H"]["std"])).square().mean()
        lb = ((b_observation(model, b["x"][:2])[:, 30:] - b["target"][:2, 30:]) / torch.tensor(norm["B"]["std"])).square().mean()
        logits = model(x, observed_counts=y).logits
        from training.mechanistic_retina.losses import expected_bernoulli_nll
        lr = expected_bernoulli_nll(logits, y, r["mask"][:2])
        return lh, lb, lr
    for loss in losses():
        loss.backward()
    gradients = {k: v.grad.clone() for k, v in model.named_parameters() if v.requires_grad}
    model.zero_grad()
    sum(losses()).backward()
    for key, parameter in model.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter.grad, gradients[key], rtol=2e-5, atol=2e-6)
    checks["macro_step_gradient_accumulation_matches_sum"] = True
    assert state_sha(teacher.state_dict()) == before
    checks["teacher_unchanged"] = True
    manifest = read_json(OUT / "dataset_manifest.json")
    assert len({r["x_sha256"] for r in manifest.values()}) == len(manifest)
    for key in ("sequence_sha256", "scored_stimulus_sha256"):
        sequence_hashes = [h for row in manifest.values() for h in row[key]]
        assert len(set(sequence_hashes)) == len(sequence_hashes) == 576
    checks["all576_sequences_unique_across_training_heldout_mechanism_full_and_scored"] = True
    assert all(set(row["keys"]) == {"x", "mask", "target"} for name, row in manifest.items() if name.startswith("D_"))
    assert all(set(row["keys"]) == {"x", "mask"} for name, row in manifest.items() if name.startswith("mechanism"))
    checks["dataset_file_separation_no_probability_or_intervention_training_targets"] = True
    result = {"status": "PASS", "verified_utc": utc(), "protocol_sha256": sha(OUT / "protocol.json"),
              "checks": checks, "optimizer_updates": 0, "review": "executor self-check, not independent audit"}
    write_json(OUT / "verification/pretraining.json", result)
    print(result)


if __name__ == "__main__":
    configure_runtime()
    verify()
