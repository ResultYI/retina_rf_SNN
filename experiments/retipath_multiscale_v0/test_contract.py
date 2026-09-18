from __future__ import annotations

import ast
from dataclasses import replace
import inspect
import math
import textwrap
import unittest

import torch

from .circuit import LocalRetipathV0, PARAMETERS, TrainingCondition, active_parameter_names, area_integral_weights
from .contracts import (
    DT_MS, Intervention, InterventionSpec, ObservationBatch, StimulusBatch,
    observation_loss, regular_pixel_bounds, select_observation,
)


def gaussian_pixel_average(bounds: torch.Tensor, sigma: float) -> torch.Tensor:
    integrals = sigma * math.sqrt(math.pi / 2) * (
        torch.erf(bounds[..., 1] / (math.sqrt(2) * sigma))
        - torch.erf(bounds[..., 0] / (math.sqrt(2) * sigma))
    )
    return (integrals / (bounds[..., 1] - bounds[..., 0])).prod(-1)


def fixture(image: torch.Tensor, bounds: torch.Tensor, *, time_count: int = 24) -> StimulusBatch:
    profile = image.new_tensor((0.4, 0.25, -0.35, 0.1, -0.2, 0.3)).repeat((time_count + 5) // 6)[:time_count]
    values = (profile[:, None] * image[None])[None, ..., None]
    return StimulusBatch(
        values, bounds, (bounds[..., 1] - bounds[..., 0]).prod(-1),
        (torch.arange(time_count, dtype=image.dtype) + 0.5) * DT_MS,
        torch.ones_like(values, dtype=torch.bool), ("hand-fixture",), ("g1-only",),
    )


def event_fixture(stimulus: StimulusBatch) -> torch.Tensor:
    events = stimulus.values.new_zeros((*stimulus.values.shape[:2], 1))
    events[:, 2::7] = 1
    return events


class PrototypeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.model = LocalRetipathV0(dtype=torch.float64)
        self.original = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        self.bounds = regular_pixel_bounds(32, dtype=torch.float64)
        self.stimulus = fixture(gaussian_pixel_average(self.bounds, 0.30), self.bounds)
        self.events = event_fixture(self.stimulus)

    def tearDown(self) -> None:
        for key, original in self.original.items():
            torch.testing.assert_close(self.model.state_dict()[key], original, rtol=0, atol=0)

    def test_01_physical_grid_integrals_and_validation(self) -> None:
        bounds64 = regular_pixel_bounds(64, dtype=torch.float64)
        center = self.bounds.mean(-1)
        square32 = ((center.abs() < 0.25).all(-1)).to(torch.float64)
        square64 = square32.reshape(32, 32).repeat_interleave(2, 0).repeat_interleave(2, 1).flatten()
        a, b = fixture(square32, self.bounds), fixture(square64, bounds64)
        with torch.no_grad():
            ta = self.model(a, observed_events=self.events)
            tb = self.model(b, observed_events=self.events)
            torch.testing.assert_close(ta.inputs["x"], tb.inputs["x"], rtol=0, atol=5e-13)
            torch.testing.assert_close(ta.outputs["p"], tb.outputs["p"], rtol=0, atol=5e-13)
            for bounds in (self.bounds, bounds64):
                constant = fixture(torch.ones(bounds.shape[0], dtype=bounds.dtype), bounds)
                trace = self.model(constant, observed_events=self.events)
                torch.testing.assert_close(trace.inputs["x"][0, :, 0, 0], constant.values[0, :, 0, 0], rtol=0, atol=5e-13)
            smooth64 = fixture(gaussian_pixel_average(bounds64, 0.30), bounds64)
            s32 = self.model(self.stimulus, observed_events=self.events)
            s64 = self.model(smooth64, observed_events=self.events)
            apertures = torch.stack((self.model.input_xy - 0.05, self.model.input_xy + 0.05), -1)
            exact = gaussian_pixel_average(apertures, 0.30)
            errors = []
            for bounds in (self.bounds, bounds64):
                actual = area_integral_weights(bounds, self.model.input_xy) @ gaussian_pixel_average(bounds, 0.30)
                errors.append(float((actual - exact).abs().max()))
            self.assertLess(errors[1], errors[0])
            response_difference = float((s32.outputs["p"] - s64.outputs["p"]).abs().max())
            self.assertLess(response_difference, 1e-4)
        bad_mask = a.input_valid.clone()
        bad_mask[..., 0, :] = False
        with self.assertRaisesRegex(ValueError, "missing input"):
            self.model(replace(a, input_valid=bad_mask), observed_events=self.events)
        with self.assertRaisesRegex(ValueError, "fully cover"):
            small = self.bounds * 0.2
            self.model(replace(a, pixel_bounds=small, pixel_area=(small[..., 1] - small[..., 0]).prod(-1)), observed_events=self.events)
        duplicate = self.bounds.clone()
        duplicate[1] = duplicate[0]
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.model(replace(a, pixel_bounds=duplicate), observed_events=self.events)
        with self.assertRaisesRegex(ValueError, "pixel_area"):
            self.model(replace(a, pixel_area=a.pixel_area * 2), observed_events=self.events)
        print(f"GRID exact_delta_p={float((ta.outputs['p']-tb.outputs['p']).abs().max()):.12g}; Gaussian node errors32/64={errors}; delta_p={response_difference:.12g}")

    def test_02_physical_scale_and_irrecoverable_pooling(self) -> None:
        bounds64 = regular_pixel_bounds(64, dtype=torch.float64)
        with torch.no_grad():
            small = fixture(gaussian_pixel_average(bounds64, 0.15), bounds64)
            large = fixture(gaussian_pixel_average(bounds64, 0.60), bounds64)
            ts = self.model(small, observed_events=self.events)
            tl = self.model(large, observed_events=self.events)
            scale_delta = float((ts.outputs["p"] - tl.outputs["p"]).abs().max())
            self.assertGreater(scale_delta, 1e-8)
            stripes = torch.where(torch.arange(64) % 2 == 0, 1.0, -1.0).to(torch.float64).expand(64, 64)
            pooled = stripes.reshape(32, 2, 32, 2).mean((1, 3))
            self.assertEqual(float(pooled.abs().max()), 0.0)
            coarse = fixture(pooled.flatten(), self.bounds)
            zero = fixture(torch.zeros(1024, dtype=torch.float64), self.bounds)
            fine = fixture(stripes.flatten(), bounds64)
            upsampled = fixture(pooled.repeat_interleave(2, 0).repeat_interleave(2, 1).flatten(), bounds64)
            tc = self.model(coarse, observed_events=self.events)
            tz = self.model(zero, observed_events=self.events)
            tf = self.model(fine, observed_events=self.events)
            tu = self.model(upsampled, observed_events=self.events)
            torch.testing.assert_close(tc.outputs["p"], tz.outputs["p"], rtol=0, atol=0)
            torch.testing.assert_close(tu.outputs["p"], tz.outputs["p"], rtol=0, atol=0)
            self.assertGreater(float(tf.inputs["x"].abs().max()), 1e-8)
            lost_delta = float((tf.outputs["p"] - tu.outputs["p"]).abs().max())
            self.assertGreater(lost_delta, 1e-8)
        print(f"SCALE max_delta_p={scale_delta:.12g}; POOL fine_vs_reconstructed_delta_p={lost_delta:.12g}; pooled_collision=exact")

    def test_03_state_output_and_direct_block(self) -> None:
        with torch.no_grad():
            normal = self.model(self.stimulus, observed_events=self.events)
            blocked = self.model(self.stimulus, observed_events=self.events,
                                 intervention=InterventionSpec(Intervention.BLOCK_DIRECT_BC_DRIVE))
            theta = self.model.physical_parameters()
            torch.testing.assert_close(normal.states["h"][:, 0], torch.zeros_like(normal.states["h"][:, 0]))
            torch.testing.assert_close(normal.states["h"][:, 1],
                                       (1 - torch.exp(-DT_MS / theta["tau_h"])) * normal.inputs["u_H"][:, 0])
            torch.testing.assert_close(normal.states["q_f"][:, 0],
                                       (1 - torch.exp(-DT_MS / theta["tau_f"])) * normal.inputs["u_B"][:, 0])
            state, output = normal.states["s_B"], normal.outputs["o_B"]
            negative = state < 0
            self.assertTrue(bool(negative.any()))
            torch.testing.assert_close(output[negative], theta["alpha"] * state[negative])
            self.assertNotEqual(state.data_ptr(), output.data_ptr())
            for name in ("x", "u_H", "u_B", "u_A"):
                torch.testing.assert_close(normal.inputs[name], blocked.inputs[name], rtol=0, atol=0)
            for name in ("h", "q_f", "q_s", "s_B", "a_AC", "q"):
                torch.testing.assert_close(normal.states[name], blocked.states[name], rtol=0, atol=0)
            for name in ("f", "o_B", "j_I", "uI", "gI"):
                torch.testing.assert_close(normal.outputs[name], blocked.outputs[name], rtol=0, atol=0)
            self.assertEqual(float(blocked.outputs["d_E"].abs().max()), 0.0)
            self.assertEqual(float(blocked.outputs["uE"].abs().max()), 0.0)
            torch.testing.assert_close(blocked.outputs["gE"], torch.ones_like(blocked.outputs["gE"]), rtol=0, atol=1e-15)
            self.assertGreater(float(normal.outputs["uI"].abs().max()), 0.0)
            torch.testing.assert_close(normal.outputs["uI"], -normal.outputs["j_I"].sum(-2, keepdim=True))
            negative_input = replace(self.stimulus, values=torch.full_like(self.stimulus.values, -0.3))
            negative_trace = self.model(negative_input, observed_events=self.events)
            self.assertTrue(bool((negative_trace.outputs["uI"] < 0).any()))
            torch.testing.assert_close(negative_trace.outputs["uI"], -negative_trace.outputs["j_I"].sum(-2, keepdim=True))
            v = [2 / 9, 2 / 9]
            z = 0.0
            expected = []
            for t in range(self.stimulus.values.shape[1]):
                for k in range(2):
                    gi = float(blocked.outputs["gI"][0, t, 0, k])
                    total = 2 + gi
                    equilibrium = (1 - gi / 3) / total
                    v[k] += (1 - math.exp(-DT_MS * total / 60)) * (equilibrium - v[k])
                m = sum(v) / 2 - 2 / 9
                z = math.exp(-DT_MS / 120) * z + (1 - math.exp(-DT_MS / 120)) * m
                expected.append(8 * m - 0.5 * z - float(blocked.states["q"][0, t, 0]) + float(theta["bias"]))
            torch.testing.assert_close(blocked.outputs["ell"][0, :, 0], torch.tensor(expected, dtype=torch.float64), rtol=0, atol=2e-13)
            delta = float((normal.outputs["p"] - blocked.outputs["p"]).abs().max())
            self.assertGreater(delta, 1e-8)
            zero = replace(self.stimulus, values=torch.zeros_like(self.stimulus.values))
            resting = self.model(zero, observed_events=torch.zeros_like(self.events))
            torch.testing.assert_close(resting.states["V"], torch.full_like(resting.states["V"], 2 / 9), rtol=0, atol=1e-15)
            self.assertEqual(sum(p.numel() for p in self.model.parameters()), 12)
        print(f"STATE separation=verified; BLOCK preserved_BC_AC_history=exact; max_delta_p={delta:.12g}; independent_tail_replay=verified")

    def test_04_causality_and_sequence_reset(self) -> None:
        normal = self.model(self.stimulus, observed_events=self.events)
        changed_values = self.stimulus.values.clone()
        changed_values[:, 8:] *= -3
        changed_events = self.events.clone()
        changed_events[:, 7:] = 1 - changed_events[:, 7:]
        future = self.model(replace(self.stimulus, values=changed_values), observed_events=changed_events)
        for values, other in ((normal.inputs, future.inputs), (normal.states, future.states), (normal.outputs, future.outputs)):
            for key in values:
                torch.testing.assert_close(values[key][:, :8], other[key][:, :8], rtol=0, atol=1e-13)
        differentiable_values = self.stimulus.values.clone().requires_grad_()
        differentiable_events = self.events.clone().requires_grad_()
        trace = self.model(replace(self.stimulus, values=differentiable_values), observed_events=differentiable_events)
        gx, gy = torch.autograd.grad(trace.outputs["ell"][0, 7, 0], (differentiable_values, differentiable_events))
        self.assertEqual(float(gx[:, 8:].abs().max()), 0.0)
        self.assertEqual(float(gy[:, 7:].abs().max()), 0.0)
        self.assertGreater(float(gx[:, :8].abs().max()), 0.0)
        self.assertGreater(float(gy[:, :7].abs().max()), 0.0)
        short = replace(self.stimulus, values=self.stimulus.values[:, :8], input_valid=self.stimulus.input_valid[:, :8], time_ms=self.stimulus.time_ms[:8])
        truncated = self.model(short, observed_events=self.events[:, :8])
        torch.testing.assert_close(truncated.outputs["p"], normal.outputs["p"][:, :8], rtol=0, atol=1e-13)
        again = self.model(self.stimulus, observed_events=self.events)
        torch.testing.assert_close(again.outputs["p"], normal.outputs["p"], rtol=0, atol=0)
        batch = replace(self.stimulus, values=torch.cat((self.stimulus.values, -self.stimulus.values)),
                        input_valid=self.stimulus.input_valid.repeat(2, 1, 1, 1),
                        sequence_id=("first", "second"), split_group_id=("g1-a", "g1-b"))
        two = self.model(batch, observed_events=self.events.repeat(2, 1, 1))
        torch.testing.assert_close(two.outputs["p"][:1], normal.outputs["p"], rtol=0, atol=1e-13)
        print("CAUSAL future_stimulus_gradient=0; current_and_future_event_gradient=0; prefix/reset/batch_isolation=verified")

    def test_05_observation_gradients_parameter_groups_and_id_independence(self) -> None:
        trace = self.model(self.stimulus, observed_events=self.events)
        names = tuple(self.model.raw)
        parameters = tuple(self.model.raw.values())
        hc_nodes = torch.tensor((12, 10, 14, 2, 22))
        bc_nodes = torch.tensor((12, 6, 8, 16, 18))
        routes = {}
        for layer, variable, nodes, branches, expected in (
            ("HC", "h", hc_nodes, torch.tensor((0,)), {"tau_h"}),
            ("BC", "o_B", bc_nodes, torch.tensor((0, 1)), {"tau_h", "a_h", "tau_f", "tau_s", "alpha"}),
            ("RGC", "events", torch.tensor((0,)), torch.tensor((0,)), set(PARAMETERS)),
        ):
            shape = (1, 24, len(nodes), len(branches))
            target = torch.zeros(shape, dtype=torch.float64)
            observation = ObservationBatch(layer, variable, nodes, branches, target, torch.ones(shape, dtype=torch.bool),
                                           torch.arange(24), "binary_occupancy" if layer == "RGC" else "synthetic_effective",
                                           "bernoulli" if layer == "RGC" else "gaussian", None if layer == "RGC" else 0.03,
                                           self.stimulus.sequence_id, self.stimulus.split_group_id)
            selected = select_observation(trace, observation)
            self.assertEqual(selected.shape, shape)
            gradients = torch.autograd.grad(observation_loss(trace, observation), parameters, allow_unused=True, retain_graph=True)
            connected = {name for name, gradient in zip(names, gradients) if gradient is not None}
            nonzero = {name for name, gradient in zip(names, gradients) if gradient is not None and float(gradient.abs()) > 0}
            self.assertEqual(connected, expected)
            self.assertEqual(nonzero, expected)
            routes[layer] = sorted(nonzero)
            with self.assertRaisesRegex(ValueError, "provenance"):
                select_observation(trace, replace(observation, sequence_id=("wrong",)))
            with self.assertRaisesRegex(ValueError, "observations"):
                observation_loss(trace, replace(observation, valid=torch.zeros_like(observation.valid)))
        state_grad = torch.autograd.grad(trace.states["s_B"].square().mean(), parameters, allow_unused=True)
        self.assertEqual({n for n, g in zip(names, state_grad) if g is not None}, {"tau_h", "a_h", "tau_f", "tau_s"})
        for condition in TrainingCondition:
            for step in (1, 30, 31, 60, 61, 180):
                expected_names = active_parameter_names(condition, step)
                groups = self.model.parameter_groups(condition, step)
                grouped_ids = [id(p) for values in groups.values() for p in values]
                self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
                self.assertEqual(set(grouped_ids), {id(self.model.raw[n]) for n in expected_names})
                self.model.configure_trainable(condition, step)
                self.assertEqual({n for n, p in self.model.raw.items() if p.requires_grad}, set(expected_names))
                unchanged = self.model(self.stimulus, observed_events=self.events)
                torch.testing.assert_close(unchanged.outputs["p"], trace.outputs["p"], rtol=0, atol=0)
        self.assertEqual(active_parameter_names("progressive", 30), ("tau_h",))
        self.assertEqual(set(active_parameter_names("progressive", 31)), {"a_h", "tau_f", "tau_s", "alpha"})
        self.assertEqual(set(active_parameter_names("progressive", 61)), set(PARAMETERS))
        renamed = replace(self.stimulus, sequence_id=("unseen-sequence",), split_group_id=("unseen-split",),
                          metadata={"dataset_id": "different-dataset", "session_id": "different-session"})
        renamed_trace = self.model(renamed, observed_events=self.events)
        for collection in ("inputs", "states", "outputs"):
            for name, value in getattr(trace, collection).items():
                torch.testing.assert_close(value, getattr(renamed_trace, collection)[name], rtol=0, atol=0)
        syntax = ast.parse(textwrap.dedent(inspect.getsource(LocalRetipathV0.forward)))
        self.assertFalse(any(isinstance(n, ast.Attribute) and n.attr == "metadata" for n in ast.walk(syntax)))
        single = LocalRetipathV0(seed=2026091801)
        paired = LocalRetipathV0(seed=2026091801)
        for name, value in single.state_dict().items():
            torch.testing.assert_close(value, paired.state_dict()[name], rtol=0, atol=0)
        bounds32 = regular_pixel_bounds(32)
        stimulus32 = fixture(gaussian_pixel_average(bounds32, 0.30), bounds32)
        trace32 = single(stimulus32, observed_events=event_fixture(stimulus32))
        gradients32 = torch.autograd.grad(trace32.outputs["ell"].square().mean(), tuple(single.parameters()))
        self.assertTrue(all(bool(torch.isfinite(g).all()) for g in gradients32))
        print(f"GRADIENT routes={routes}; BC_state_excludes_alpha=verified; parameter_counts_C=1/4/12; IDs_do_not_select_mechanism=verified")
        print("FLOAT32 forward/all_12_gradients=finite; paired_seed_initialization=exact; optimizer_steps=0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
