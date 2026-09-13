# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
# How to run: imported by calculate_metrics.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
EPS: Final = np.finfo(np.float64).eps


@dataclass(frozen=True,slots=True)
class Repeatability:
    split_r: tuple[float,...]
    r3: float
    raw_r_mean: float
    rel6: float
    ceiling: float
    status: str


def pearson(x: FloatArray,y: FloatArray) -> float:
    a=x.astype(np.float64)-np.mean(x,dtype=np.float64)
    b=y.astype(np.float64)-np.mean(y,dtype=np.float64)
    scale=np.sqrt(np.sum(a*a)*np.sum(b*b))
    return float(np.sum(a*b)/scale) if scale>0 and np.isfinite(scale) else float('nan')


def average_ranks(x: FloatArray) -> FloatArray:
    _,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    ranks=np.cumsum(counts)-(counts-1)/2
    return ranks[inverse]


def spearman(x: FloatArray,y: FloatArray) -> float:
    return pearson(average_ranks(x),average_ranks(y))


def repeatability(repeats: FloatArray,partitions: tuple[tuple[int,...],...]) -> Repeatability:
    correlations=[]
    for a in partitions:
        b=tuple(i for i in range(6) if i not in a)
        correlations.append(pearson(repeats[list(a)].mean(axis=0),repeats[list(b)].mean(axis=0)))
    values=np.array(correlations)
    raw_mean=float(values.mean())
    if not np.isfinite(values).all() or np.any(np.abs(values)>=1):
        return Repeatability(tuple(correlations),float('nan'),raw_mean,float('nan'),float('nan'),'UNDEFINED_SPLIT_FISHER')
    r3=float(np.tanh(np.arctanh(values).mean()))
    if not np.isfinite(r3) or abs(1+r3)<=EPS:
        return Repeatability(tuple(correlations),r3,raw_mean,float('nan'),float('nan'),'UNDEFINED_DENOMINATOR')
    rel6=2*r3/(1+r3)
    ceiling=float(np.sqrt(rel6)) if rel6>0 else float('nan')
    status='DEFINED'
    if rel6<=0: status='UNDEFINED_NONPOSITIVE_REL6'
    if rel6>1: status='FINITE_REPEAT_ESTIMATOR_ANOMALY'
    return Repeatability(tuple(correlations),r3,raw_mean,rel6,ceiling,status)
