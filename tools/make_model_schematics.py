"""Draw source-checked architecture figures without training or reading spike targets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from baselines.center_surround_ln import CenterSurroundLN, CONTEXT_BINS
from baselines.compact_causal_cnn import CompactCausalCNN
from models.mechanistic_retina.contracts import MechanisticRetinaConfig
from models.mechanistic_retina.retipath import RetiPath
from models.mechanistic_retina.support_partition import _AC_RADIUS, _BC_RADIUS

OUT = ROOT / "output/figures/model_schematics"
REGISTRY = ROOT / "output/evaluations/retipath_final_model_evidence_20260913/model_registry.json"
MOVIE = ROOT / "data/real/schottdorf_lee_2021_macaque/1x10_256.mpg"
INK, MUTED, EDGE = "#24282c", "#636d74", "#6b7882"
SOURCES = [
    "output/evaluations/retipath_final_model_evidence_20260913/model_registry.json",
    "output/evaluations/retipath_final_model_evidence_20260913/MIGRATION_PROTOCOL.md",
    "work/retipath_final_common.py", "work/retipath_final_prediction.py",
    "models/mechanistic_retina/retipath.py", "models/mechanistic_retina/retipath_spatial_ei.py",
    "models/mechanistic_retina/contracts.py", "models/mechanistic_retina/rgc_state.py",
    "models/mechanistic_retina/h1_pathway.py", "models/mechanistic_retina/bipolar_subunits.py",
    "models/mechanistic_retina/amacrine_pathways.py", "models/mechanistic_retina/support_partition.py",
    "models/mechanistic_retina/pathway_temporal.py", "models/mechanistic_retina/state.py",
    "models/mechanistic_retina/local_bc_nonlinearity.py", "models/mechanistic_retina/shared_subunits.py",
    "baselines/compact_causal_cnn.py", "baselines/center_surround_ln.py",
    "data/schottdorf_lee_2021.py", "data/schottdorf_lee_multirecording.py",
    "data/schottdorf_lee_catalog.py", "data/real/schottdorf_lee_2021_repository/README.md",
]


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_sources() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if registry["architecture_id"] != "retipath_spatial_conductance_v1":
        raise ValueError("Formal architecture changed: re-read the implementation before drawing.")
    checks, groups = [], Counter()
    configs = set()
    for cell, entry in registry["cells"].items():
        groups[entry["group"]] += 1
        for name in ("RetiPath", "CNN", "LN"):
            refs = entry[name].values() if name == "RetiPath" else [entry[name]]
            for ref in refs:
                path = ROOT / ref["path"]
                actual = digest(path)
                if actual != ref["sha256"]:
                    raise ValueError(f"Checkpoint identity mismatch: {path}")
                saved = torch.load(path, weights_only=True, map_location="cpu")
                state = saved["model"]
                if name == "RetiPath":
                    assert saved["backend"] == "spatial_conductance"
                    assert saved["phase"] == "refit"
                    assert list(saved["cone_positions_degs"].shape) == [289, 2]
                    assert list(saved["cell_positions_degs"].shape) == [1, 2]
                    assert list(state["feature_bank.path_spatial_basis"].shape) == [1, 4, 2, 289]
                    assert list(state["spatial_ei.rms_e"].shape) == [1, 2]
                    assert saved["trainable_parameters"] == 37
                    configs.add(json.dumps(saved["model_config"], sort_keys=True))
                elif name == "CNN":
                    assert list(state["conv1.weight"].shape) == [4, 1, 12, 5, 5]
                    assert list(state["conv2.weight"].shape) == [4, 4, 9, 3, 3]
                    assert list(state["spatial_readout"].shape) == [4, 11, 11]
                else:
                    assert saved["context_bins"] == 60
                    assert list(state["raw_temporal"].shape) == [2, 60]
                    assert list(state["grid_xy"].shape) == [17, 17, 2]
                if name != "RetiPath":
                    assert saved["history"] == {"dt_ms": 1000 / 150, "tau_ms": 30.0}
                checks.append({"model": name, "cell": cell, "path": path.relative_to(ROOT).as_posix(),
                               "sha256": actual, "verified": True})
    assert len(configs) == 1 and len(registry["cells"]) == 22
    config = json.loads(next(iter(configs)))
    assert config["dt_ms"] == 1000 / 150 and config["lag_steps"] == 16
    assert config["history_tau_ms"] == 30 and config["h1_radius_deg"] == .18
    assert CONTEXT_BINS == 60
    assert _BC_RADIUS == {"midget": .06, "parasol": .10}
    assert _AC_RADIUS == {"midget": .13, "parasol": .15}
    torch.set_num_threads(2)
    entry = registry["cells"]["67#14"]
    cp = torch.load(ROOT / entry["RetiPath"]["2026091301"]["path"], weights_only=True, map_location="cpu")
    model = RetiPath(MechanisticRetinaConfig(**cp["model_config"]), cp["cone_positions_degs"],
                    cp["cell_positions_degs"], tuple(cp["cell_types"]), tuple(cp["polarities"]),
                    rms_e=torch.tensor(cp["rms"]["e"]), rms_i=torch.tensor(cp["rms"]["i"]))
    model.load_state_dict(cp["model"], strict=True)
    model.eval().requires_grad_(False)
    x, history = torch.zeros(1, 150, 289), torch.zeros(1, 150, 1)
    shapes = {}
    with torch.inference_mode():
        parts = model.spatial_components(x)
        shapes["RetiPath"] = {"input": list(x.shape), "h1": list(parts.h1.modulated_cones.shape),
                             "direct": list(parts.direct.shape), "broad": list(parts.broad.shape),
                             "ac": list(parts.ac_states.shape), "u_e": list(parts.u_e.shape),
                             "output": list(model.forward_sequence(x, observed_counts=history).spike_probability.shape)}
        for name, cls in (("CNN", CompactCausalCNN), ("LN", CenterSurroundLN)):
            saved = torch.load(ROOT / entry[name]["path"], weights_only=True, map_location="cpu")
            baseline = cls(saved["history"]["dt_ms"], saved["history"]["tau_ms"], 61001)
            baseline.load_state_dict(saved["model"], strict=True)
            baseline.eval().requires_grad_(False)
            if name == "CNN":
                def capture(label: str):
                    def hook(module, inputs, output):
                        shapes[label] = list(output.shape)
                    return hook
                baseline.conv1.register_forward_hook(capture("CNN_conv1"))
                baseline.conv2.register_forward_hook(capture("CNN_conv2"))
            logits = baseline(x, history)
            shapes[name + "_output"] = list(torch.sigmoid(logits).shape)
    assert shapes["CNN_conv1"] == [1, 4, 150, 13, 13]
    assert shapes["CNN_conv2"] == [1, 4, 150, 11, 11]
    assert shapes["RetiPath"]["direct"] == [1, 150, 1, 2, 2]
    return {"formal_registry": REGISTRY.relative_to(ROOT).as_posix(), "config": config,
            "source_files": [{"path": p, "sha256": digest(ROOT / p)} for p in SOURCES],
            "checkpoint_checks": checks, "groups": dict(groups), "shape_probe": shapes,
            "verification": "Read-only checkpoint metadata and zero-input shape checks; no observed spike targets read.",
            "unverified": [], "cnn_scope": "CNN in the formal registry: CompactCausalCNN"}


class Canvas:
    def __init__(self, width: int, height: int) -> None:
        self.fig, self.ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, width), ylim=(height, 0), aspect="equal")
        self.ax.axis("off")
        self.width, self.height = width, height

    def text(self, x, y, label, size=10.5, color=INK, ha="center"):
        return self.ax.text(x, y, label, ha=ha, va="center", fontsize=size,
                            color=color, linespacing=1.25, zorder=10)

    def line(self, points, color=EDGE, lw=1.0, dashed=False):
        self.ax.plot(*zip(*points), color=color, lw=lw, ls=(0, (3, 3)) if dashed else "-",
                     solid_capstyle="round", zorder=2)

    def arrow(self, points, color=EDGE, dashed=False, lw=1.1):
        if len(points) > 2:
            self.line(points[:-1], color, lw, dashed)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                         mutation_scale=9, lw=lw, color=color,
                                         linestyle=(0, (3, 3)) if dashed else "-",
                                         shrinkA=0, shrinkB=0, zorder=3))

    def rect(self, x, y, w, h, color="white", edge=EDGE, lw=.85, dashed=False, zorder=4):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor=edge,
                                   lw=lw, linestyle=(0, (3, 3)) if dashed else "-", zorder=zorder))

    def circle(self, x, y, radius=13, label="", color="white", edge=EDGE):
        self.ax.add_patch(Circle((x, y), radius, facecolor=color, edgecolor=edge, lw=.9, zorder=4))
        if label:
            self.text(x, y, label)

    def dot(self, x, y):
        self.ax.add_patch(Circle((x, y), 2, facecolor=EDGE, edgecolor="none", zorder=5))

    def save(self, name):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        text_boxes = []
        for item in self.ax.texts:
            box = item.get_window_extent(renderer)
            if box.x0 < 0 or box.y0 < 0 or box.x1 > self.width or box.y1 > self.height:
                raise ValueError(f"Text outside {name}: {item.get_text()}")
            text_boxes.append({"text": item.get_text(), "xyxy": [box.x0, box.y0, box.x1, box.y1]})
        for suffix in ("svg", "png", "pdf"):
            self.fig.savefig(OUT / f"{name}.{suffix}", dpi=240, facecolor="white")
        qa = OUT / "qa"
        qa.mkdir(exist_ok=True)
        (qa / f"{name}_text_bounds.json").write_text(json.dumps(text_boxes, ensure_ascii=False, indent=2), encoding="utf-8")
        plt.close(self.fig)


E_COLOR, I_COLOR = "#3f806a", "#ad6955"
PAPER_BLUE, PAPER_GREEN, PAPER_RED = "#e9eff3", "#edf4ef", "#f6eeeb"


def frames():
    capture = cv2.VideoCapture(str(MOVIE))
    if not capture.isOpened():
        raise FileNotFoundError(MOVIE)
    images = []
    try:
        for frame in (808, 809, 810):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, bgr = capture.read()
            if not ok:
                raise ValueError("Movie frame unavailable")
            images.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    return images


def video(c, x, y, images, side=76, full=False):
    for i, im in enumerate(images):
        xx, yy = x + 10*i, y - 10*i
        c.ax.imshow(im, extent=(xx, xx+side, yy+side, yy), zorder=4+i)
        c.ax.add_patch(Rectangle((xx, yy), side, side, facecolor="none", edgecolor=EDGE, lw=.7, zorder=8))
    c.text(x+(side+20)/2, y-43, "自然動画")
    c.text(x+(side+20)/2, y+side+17, r"$t-2,\;t-1,\;t$", 9)
    if full:
        c.text(x+(side+20)/2, y+side+39, "150 bins", 9, MUTED)
    return x+side+20


def grid(c, x, y, side, *, disk=None, color=E_COLOR, kernel=None):
    c.rect(x, y, side, side, "#fafbfc", "#a5b0b7", .65)
    for n in range(1, 17):
        q = side*n/17
        c.ax.plot([x+q, x+q], [y, y+side], color="#d6dde1", lw=.28, zorder=5)
        c.ax.plot([x, x+side], [y+q, y+q], color="#d6dde1", lw=.28, zorder=5)
    if disk is not None:
        c.ax.add_patch(Circle((x+side/2, y+side/2), side*disk, facecolor=color,
                              alpha=.14, edgecolor="none", zorder=6))
        c.ax.add_patch(Circle((x+side/2, y+side/2), side*disk, facecolor="none",
                              edgecolor=color, lw=1.15, zorder=7))
        c.circle(x+side/2, y+side/2, 1.8, color=color, edge=color)
    if kernel:
        kside = side*kernel/17
        c.ax.add_patch(Rectangle((x+side*.45, y+side*.26), kside, kside,
                                 facecolor="#dbe6ef", edgecolor="#56778e", lw=1, zorder=7))


def cone(c, x, y, side=70, full=False, kernel=None):
    grid(c, x, y, side, kernel=kernel)
    c.text(x+side/2, y-23, "錐体入力")
    c.text(x+side/2, y+side+19, "L+M · 17×17", 9)
    if full:
        c.text(x+side/2, y+side+41, "150 × 289", 9, MUTED)


def recurrent(c, x, y, span=38, label=False, color=EDGE):
    c.arrow([(x, y), (x, y+18), (x+span, y+18), (x+span, y)], color)
    if label:
        c.text(x+span/2, y+32, "$t-1$", 8.5, MUTED)


def h1(c, x, y, full=False):
    points = [(x-22,y), (x-9,y-18), (x+12,y-16), (x+24,y+5), (x-1,y+17)]
    for a,b in ((0,1),(0,4),(1,2),(1,4),(2,3),(2,4),(3,4)):
        c.line([points[a], points[b]], "#8d9eaa", .8)
    for xx, yy in points:
        c.circle(xx, yy, 4.8, color=PAPER_BLUE, edge="#698293")
    c.text(x, y-38, "H1")
    c.arrow([(x-1,y+22),(x-1,y+43),(x+41,y+43),(x+41,y+5),(x+30,y+5)])
    if full:
        c.text(x+20,y+56,r"$t-1$",8.5,MUTED)


def bc(c, x, y, broad=False, full=False):
    width, height = 99, 62 if full else 48
    c.rect(x, y-height/2, width, height, PAPER_GREEN, E_COLOR)
    c.text(x+width/2, y-13 if full else y, "BC broad" if broad else "BC direct", 10)
    if full:
        for offset, label in ((5, "持続"), (51, "過渡")):
            c.rect(x+offset, y+2, 43, 22, "white", "#b7cebf", .65)
            c.text(x+offset+21.5, y+13, label, 8.5)
        c.text(x+width/2, y+height/2+18, "16 lags", 8.5, MUTED)
    return width


def ac(c, x, y, full=False):
    c.text(x+32, y-63, "AC")
    for yy in (y-14,y+14):
        c.circle(x+32, yy, 10, color=PAPER_RED, edge=I_COLOR)
        c.arrow([(x, y), (x+14, yy), (x+21, yy)], I_COLOR, lw=.85)
        c.line([(x+42, yy), (x+58, y)], I_COLOR, .85)
    c.arrow([(x+32,y-24),(x+32,y-45),(x+58,y-45),(x+58,y-14),(x+43,y-14)],I_COLOR)
    c.arrow([(x+32,y+24),(x+32,y+45),(x+58,y+45),(x+58,y+14),(x+43,y+14)],I_COLOR)
    if full:
        c.text(x+45,y+62,r"$t-1$",8.5,MUTED)
    return x+58


def conductance(c, x, y, inhibitory=False, full=False):
    col = I_COLOR if inhibitory else E_COLOR
    c.rect(x, y-23, 65, 46, PAPER_RED if inhibitory else PAPER_GREEN, col)
    c.text(x+32.5, y, r"$g_{I,k}$" if inhibitory else r"$g_{E,k}$", 12, col)
    if full:
        c.text(x+32.5, y-39, "抑制性電導" if inhibitory else "興奮性電導", 9, col)
        c.text(x-28, y-15, r"$u_{I,k}$" if inhibitory else r"$u_{E,k}$", 9, col)


def membrane(c, x, y, full=False):
    c.circle(x+7, y-6, 27, color="#f2f0f6", edge="#a5a0b1")
    c.circle(x, y, 27, r"$V_k(t)$", "#f8f7fa", "#716d7d")
    c.text(x, y-47, "膜状態", 10)
    recurrent(c, x-17, y+21, 35, full)
    c.text(x+58, y+47, "K=2", 9, MUTED)


def history(c, x, y, target_x, target_y, full=False):
    c.text(x+45, y-18, "観測発火履歴", 9.5)
    c.line([(x,y+13), (x+90,y+13)], "#89939c", .75)
    for q in (8, 24, 29, 48, 70, 83):
        c.line([(x+q,y+13),(x+q,y)], "#596772", .9)
    c.arrow([(x+93,y+7),(target_x,y+7),(target_x,target_y)], "#7a8690", True)
    if full:
        c.text(x+45, y+33, "1 bin遅延 · 30 ms", 8.5, MUTED)


def sigmoid(c, x, y):
    c.circle(x, y, 23)
    t = np.linspace(-3,3,60)
    c.ax.plot(x+t*5.5, y-(1/(1+np.exp(-t))-.5)*30, color=INK, lw=1.15, zorder=6)
    c.text(x, y-38, "sigmoid", 9.5)


def probability(c, x, y, full=False):
    c.text(x+58, y-35, "単一RGC", 10)
    for j, label in enumerate(("$p_0$", "$p_1$", r"$\cdots$", "$p_t$")):
        c.rect(x+j*29, y-17, 29, 34, "#fafbfc", "#9da9b0", .6)
        c.text(x+j*29+14.5, y, label, 10)
    c.text(x+58, y+36, "発火確率", 9.5)
    if full:
        c.text(x+58, y+57, "150 × 1", 9, MUTED)


def draw_retipath(images, full):
    c = Canvas(1730 if full else 1530, 420 if full else 350)
    y = 220 if full else 175
    video_end = video(c, 18, y-37, images, 73, full)
    cone_x, cone_side = 168, 67
    cone(c, cone_x, y-cone_side/2, cone_side, full)
    c.arrow([(video_end+7,y),(cone_x-10,y)])
    if full:
        c.text(141, y-19, "校正", 8.5, MUTED)
        c.arrow([(cone_x+cone_side+4,y),(400,y)])
        c.dot(273,y)
        c.arrow([(273,y),(273,111),(315,111)])
        h1(c, 343,111, False)
        c.arrow([(371,111),(413,111),(413,y-14)])
        c.circle(413,y,13,"+")
        c.text(403,y-27,"−",10)
        fork, support_x, bcx, acx, gx, vx = 457, 492, 625, 808, 968, 1130
        top, bottom = 130, 322
        readx, sigx, outx = 1290, 1468, 1577
        c.arrow([(427,y),(fork,y)])
    else:
        c.arrow([(cone_x+cone_side+4,y),(278,y)])
        h1(c, 309,y)
        fork, support_x, bcx, acx, gx, vx = 362, 394, 503, 657, 810, 957
        top, bottom = 104, 255
        readx, sigx, outx = 1110, 1290, 1390
        c.arrow([(340,y),(fork,y)])
    c.dot(fork,y)
    side=73 if full else 63
    for yy, disk, label, broad in ((top,.22,"局所",False),(bottom,.43,"広域",True)):
        c.arrow([(fork,y),(fork,yy),(support_x-7,yy)])
        grid(c,support_x,yy-side/2,side,disk=disk)
        c.text(support_x+side/2,yy-side/2-21,label,10,E_COLOR)
        c.arrow([(support_x+side+5,yy),(bcx-8,yy)],E_COLOR)
        bc(c,bcx,yy,broad,full)
    c.arrow([(bcx+105,top),(gx-8,top)],E_COLOR)
    c.arrow([(bcx+105,bottom),(acx,bottom)],E_COLOR)
    ac_end=ac(c,acx,bottom,full)
    c.arrow([(ac_end+3,bottom),(gx-8,bottom)],I_COLOR)
    for yy, inh in ((top,False),(bottom,True)):
        conductance(c,gx,yy,inh,full)
        c.arrow([(gx+70,yy),(vx-64,yy),(vx-23,y+(-15 if not inh else 15))],I_COLOR if inh else E_COLOR)
    membrane(c,vx,y,full)
    avgx = vx+85
    c.arrow([(vx+34,y),(avgx-19,y)])
    c.circle(avgx,y,18,r"$\langle\cdot\rangle$",color="white")
    c.text(avgx,y-36,"空間平均" if full else "平均",9.5)
    c.arrow([(avgx+19,y),(readx-8,y)])
    c.rect(readx,y-26,115,52,"white")
    c.text(readx+57.5,y,"RGC 読み出し",10)
    hx=readx-120
    history(c,hx,42 if full else 34,readx+57.5,y-27,full)
    adapt_y=y+105 if full else y+103
    tapx=readx-20
    c.dot(tapx,y)
    c.arrow([(tapx,y),(tapx,adapt_y),(readx+8,adapt_y)])
    c.rect(readx+8,adapt_y-17,95,34,"#f8f8f9")
    c.text(readx+55.5,adapt_y,"順応",9.5)
    c.arrow([(readx+83,adapt_y-17),(readx+83,y+27)])
    c.text(readx+96,y+43,"−",10,MUTED)
    c.arrow([(readx+119,y),(sigx-25,y)])
    c.text((readx+115+sigx-25)/2,y+21,"logit",8.5,MUTED)
    sigmoid(c,sigx,y)
    c.arrow([(sigx+26,y),(outx-7,y)])
    probability(c,outx,y,full)
    c.save("retipath_full" if full else "retipath_clean")


def feature_maps(c,x,y,side,shape,full=False,kernel=None):
    colors=("#f5f8fa","#eef3f7","#e6eef3","#e0eaf1")
    for j in range(4):
        xx,yy=x+j*6,y-j*6
        c.rect(xx,yy,side,side,colors[j],"#7993a5",.75,zorder=4+j*2)
        spatial_side = 13 if kernel else 11
        for k in range(1,spatial_side):
            q=side*k/spatial_side
            c.ax.plot([xx+q,xx+q],[yy,yy+side],lw=.28,color="#c7d5df",zorder=5+j*2)
            c.ax.plot([xx,xx+side],[yy+q,yy+q],lw=.28,color="#c7d5df",zorder=5+j*2)
    if kernel:
        kside=side*kernel/13
        c.ax.add_patch(Rectangle((x+18+side*.43,y-18+side*.28),kside,kside,
                                 facecolor="#c5d7e4",edgecolor="#4e738c",lw=.85,zorder=13))
    c.text(x+(side+18)/2,y+side+23,shape,9.5)
    if full:
        c.text(x+(side+18)/2,y+side+46,"4 channels",9,MUTED)
    return x+side+18


def conv_connection(c,left,right,y,label,detail,kernel_from):
    c.arrow([(left,y),(right,y)],"#607d91")
    c.text((left+right)/2,y-71,"畳み込み + ReLU",9.5)
    c.text((left+right)/2,y-47,label,9.5)
    if detail:
        c.text((left+right)/2,y+66,detail,8.5,MUTED)
    kx,ky=kernel_from
    c.line([(kx,ky),(right,y-34)],"#a4b7c4",.65)
    c.line([(kx,ky+18),(right,y+30)],"#a4b7c4",.65)


def draw_cnn(images,full):
    c=Canvas(1480 if full else 1370,320 if full else 280)
    y=171 if full else 152
    end=video(c,18,y-37,images,73,full)
    cone(c,170,y-38,76,full,kernel=5)
    c.arrow([(end+6,y),(163,y)])
    f1x,f2x=455,750 if full else 691
    f1side,f2side=92,77
    conv_connection(c,254,f1x-9,y,"12×5×5","左pad 11" if full else "",(170+76*.45+76*5/17,y-38+76*.26))
    f1end=feature_maps(c,f1x,y-f1side/2,f1side,"4×13×13" if not full else "4×150×13×13",full,kernel=3)
    conv_connection(c,f1end+7,f2x-9,y,"9×3×3","時間dilation 6 · 左pad 48" if full else "時間dilation 6",(f1x+18+f1side*.43+f1side*3/13,y-f1side/2-18+f1side*.28))
    f2end=feature_maps(c,f2x,y-f2side/2,f2side,"4×11×11" if not full else "4×150×11×11",full)
    sumx=1010 if full else 925
    for offset in (-28,-10,10,28):
        c.line([(f2end+4,y+offset),(sumx-21,y)],"#9aaab6",.7)
    c.arrow([(f2end+8,y),(sumx-19,y)])
    c.circle(sumx,y,18,r"$\Sigma$")
    c.text((f2end+sumx)/2,y-56,"空間読み出し",9.5)
    plusx=sumx+112
    c.arrow([(sumx+20,y),(plusx-15,y)])
    c.circle(plusx,y,14,"+")
    history(c,sumx-24,37 if full else 29,plusx,y-15,full)
    c.circle(plusx,y+76,10,"b")
    c.arrow([(plusx,y+65),(plusx,y+16)])
    sigx=plusx+99
    c.arrow([(plusx+17,y),(sigx-26,y)])
    c.text((plusx+sigx)/2,y+22,"logit",8.5,MUTED)
    sigmoid(c,sigx,y)
    outx=sigx+89
    c.arrow([(sigx+26,y),(outx-7,y)])
    probability(c,outx,y,full)
    if full:
        c.text(560,287,"60 bins",9,MUTED)
        c.line([(267,263),(267,272),(853,272),(853,263)],"#a8b4bd",.65)
    else:
        c.text(208,y+97,"60 bins",9,MUTED)
    c.save("cnn_full" if full else "cnn_clean")


def gaussian(c,x,y,side,broad=False):
    c.rect(x,y,side,side,"#fbfbfb","#a2abb1",.6)
    radius=side*(.43 if broad else .27)
    for fraction,shade in ((1,"#e7edf1"),(.68,"#c5d2dc"),(.35,"#9cafbe")):
        c.ax.add_patch(Circle((x+side/2,y+side/2),radius*fraction,facecolor=shade,
                              edgecolor="none",zorder=5))


def temporal_filter(c,x,y,w,label):
    c.rect(x,y-15,w,30,"#f4f6f8","#8c9daa",.75)
    for j in range(1,6):
        c.line([(x+w*j/6,y-15),(x+w*j/6,y+15)],"#b2c1cc",.5)
    c.text(x+w/2,y,label,10)


def draw_ln(images,full):
    c=Canvas(1170 if full else 1100,330 if full else 276)
    y=180 if full else 150
    end=video(c,17,y-34,images,70,full)
    cone(c,159,y-33,66,full)
    c.arrow([(end+6,y),(151,y)])
    left, fw = 298, 342 if full else 271
    top,bottom=y-64,y+64
    c.rect(left,top-45,fw,218,"none","#b7c0c6",.7,zorder=1)
    c.text(left+fw/2,top-22,"時空間線形フィルタ",10)
    fork=left+15
    c.arrow([(232,y),(fork,y)])
    c.dot(fork,y)
    gx=left+39
    tx=left+157 if full else left+130
    tw=105 if full else 84
    side=49
    for yy,broad,lab in ((top+18,False,"中心"),(bottom-12,True,"周辺")):
        c.arrow([(fork,y),(fork,yy),(gx-5,yy)])
        gaussian(c,gx,yy-side/2,side,broad)
        c.text(gx+side/2,yy+side/2+16,lab,8.5,MUTED)
        c.arrow([(gx+side+4,yy),(tx-6,yy)])
        temporal_filter(c,tx,yy,tw,r"$h_s(\tau)$" if broad else r"$h_c(\tau)$")
    c.text(tx+tw/2,y+96,"60 bins",9,MUTED)
    sumx=left+fw+61
    c.arrow([(tx+tw+4,top+18),(sumx,top+18),(sumx,y-18)])
    c.arrow([(tx+tw+4,bottom-12),(sumx,bottom-12),(sumx,y+18)])
    c.text(sumx-13,y-30,"+",10)
    c.text(sumx-13,y+30,"−",10)
    c.circle(sumx,y,17,r"$\Sigma$")
    plusx=sumx+111
    c.arrow([(sumx+20,y),(plusx-16,y)])
    c.circle(plusx,y,14,"+")
    history(c,sumx-17,35 if full else 28,plusx,y-15,full)
    c.circle(plusx,y+75,10,"b")
    c.arrow([(plusx,y+64),(plusx,y+16)])
    sigx=plusx+109
    c.arrow([(plusx+17,y),(sigx-26,y)])
    c.text((plusx+sigx)/2,y+22,"logit",8.5,MUTED)
    sigmoid(c,sigx,y)
    outx=sigx+90
    c.arrow([(sigx+26,y),(outx-7,y)])
    probability(c,outx,y,full)
    c.save("ln_full" if full else "ln_clean")


def source_evidence(recheck):
    saved_path=OUT / "verified_architecture.json"
    if recheck or not saved_path.is_file():
        evidence=verify_sources()
    else:
        evidence=json.loads(saved_path.read_text(encoding="utf-8"))
        changed=[s["path"] for s in evidence["source_files"] if digest(ROOT/s["path"])!=s["sha256"]]
        if changed:
            raise ValueError(f"Sources changed; re-read before drawing: {changed}")
        for check in evidence["checkpoint_checks"]:
            if digest(ROOT/check["path"]) != check["sha256"]:
                raise ValueError(f"Checkpoint changed: {check['path']}")
    lock=json.loads((REGISTRY.parent/"source_lock.json").read_text(encoding="utf-8"))
    movie_hash=digest(MOVIE)
    assert movie_hash==lock["frozen_sha256"][str(MOVIE.relative_to(ROOT))]
    evidence.update(movie_path=MOVIE.relative_to(ROOT).as_posix(), movie_sha256=movie_hash,
                    movie_frame_indices_zero_based=[808,809,810],
                    movie_thumbnail_note="Three consecutive RGB movie frames; all other graphics are structural symbols, not fitted activations or predicted spike data.")
    saved_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only",choices=("all","retipath","cnn","ln"),default="all")
    parser.add_argument("--recheck",action="store_true",help="Repeat frozen-checkpoint and zero-input shape checks.")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    font=Path("C:/Windows/Fonts/YuGothM.ttc")
    if not font.is_file():
        raise FileNotFoundError("Yu Gothic font missing; choose an installed Japanese TrueType font.")
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({"font.family":font_manager.FontProperties(fname=font).get_name(),
                         "pdf.fonttype":42,"svg.fonttype":"none","axes.unicode_minus":False,
                         "mathtext.fontset":"dejavusans","font.size":10.5})
    evidence=source_evidence(args.recheck)
    images=frames()
    for key,draw in (("retipath",draw_retipath),("cnn",draw_cnn),("ln",draw_ln)):
        if args.only in ("all",key):
            for full in (True,False):
                draw(images,full)
    print("Architecture sources unchanged;",len(evidence["checkpoint_checks"]),"frozen checkpoint checks available.")
    print("Figures:",OUT)


if __name__=="__main__":
    main()
