# -*- coding: utf-8 -*-
"""Interval specs -- the second way a user may state a critical-speed target.

  point      '1=3680.8'          base band [3680.8, 3680.8]
  interval   '1=[10000,12000]'   base band [10000, 12000]
  one-sided  '1=[14400,+]'       base band [14400, +inf)   ('+' / 'inf' / empty)

A *relaxation* r widens a band symmetrically in relative terms when judging:
in band means inside [lo*(1-r), hi*(1+r)].  For a point band that is exactly the
old "+-r %" rule; for a user interval, r = 0 is literally the range they asked
for, and r > 0 answers "how far outside my range am I willing to look".

The denoiser is conditioned on the two edges (see effective() for how a band
wider than the model can fill is brought back inside its range), so nothing here
needs to know about the network.

Everything is per order, 1..3.
"""
import math

import numpy as np

INF_CAP = 1e12


# --------------------------------------------------------------------------- #
# parsing
# --------------------------------------------------------------------------- #
def split_specs(text):
    """Split on commas that are not inside brackets: '1=[10,12],2=5000' -> 2."""
    out, depth, buf = [], 0, []
    for ch in str(text).replace(";", ","):
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth <= 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [token.strip() for token in out if token.strip()]


def _pair(body):
    for sep in ("~", "..", ","):
        if sep in body:
            low, _, high = body.partition(sep)
            return low.strip(), high.strip()
    return body.strip(), body.strip()


def _bound(text, default):
    text = str(text).strip().lower()
    if text in ("", "+", "inf", "infinity", "max", "none"):
        return float(default)
    return float(text)


def parse_specs(text):
    """'1=5000' / '1=[10000,12000]' / '1=[14400,+]' -> {1: (lo, hi)}."""
    out = {}
    for token in split_specs(text):
        key, _, value = token.partition("=")
        order = int(key.strip().lower().replace("cs", "").replace("_", ""))
        value = value.strip()
        if value[:1] in "[(" or value[-1:] in "])":
            low, high = _pair(value.strip("[]() "))
            lo = _bound(low, 0.0)
            hi = _bound(high, INF_CAP)
        else:
            lo = hi = float(value)
        if lo <= 0.0:
            raise ValueError("order %d: lower bound must be positive" % order)
        if hi < lo:
            lo, hi = hi, lo
        out[order] = (lo, hi)
    return out


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #
def centre(lo, hi):
    """Geometric centre; for an open upper bound the lower bound is the anchor."""
    if hi >= INF_CAP:
        return max(lo, 1e-9)
    return math.sqrt(max(lo, 1e-9) * hi)


def log_half(lo, hi):
    if hi >= INF_CAP or hi <= lo:
        return 0.0 if hi <= lo else float("inf")
    return 0.5 * math.log(hi / lo)


def effective(lo, hi, h_max):
    """The band the network is asked to fill, after capping its log width.

    A denoiser cannot spread mass over an arbitrarily wide band, so anything
    wider than exp(h_max) is brought back in.  A *bounded* band is shrunk around
    its geometric centre (symmetric, so the model still fills the middle of what
    the user asked); an *open* band (one-sided spec) is anchored at its lower
    bound, because there the interesting edge is the one the user named.
    """
    if hi <= lo:
        return lo, lo
    if math.log(hi / lo) <= h_max:
        return lo, hi
    if hi >= INF_CAP:
        return lo, lo * math.exp(h_max)
    c = centre(lo, hi)
    return c * math.exp(-0.5 * h_max), c * math.exp(0.5 * h_max)


def band_text(lo, hi, digits=0):
    return "[%.*f, %.*f]" % (digits, lo, digits, min(hi, INF_CAP))


# --------------------------------------------------------------------------- #
# judging
# --------------------------------------------------------------------------- #
def inside(values, lo, hi, relax=0.0):
    """Inside [lo*(1-relax), hi*(1+relax)]."""
    values = np_asarray(values)
    low = lo * (1.0 - relax)
    high = hi * (1.0 + relax)
    return (values >= low) & (values <= high)


def deviation(values, lo, hi):
    """Signed relative distance from the band: 0 inside, +above, -below.

    Sign and magnitude are both meaningful (it is what the ranking sorts on), so
    a value 3 % under the band and one 3 % over are equally wrong but on
    opposite sides; for a point band this degenerates to (v - lo)/lo.
    """
    values = np_asarray(values).astype(float)
    scale = max(centre(lo, hi), 1e-9)      # never INF_CAP: see centre()
    above = np.maximum(values - min(hi, INF_CAP), 0.0) / scale
    below = np.maximum(lo - values, 0.0) / scale
    return above - below


def np_asarray(values):
    return np.asarray(values, dtype=np.float64)