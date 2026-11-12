"""The cost of context: attention, a recurrence and an SSM as the sequence grows.

No model is trained here; the numbers are the arithmetic of the three
mechanisms, with real shapes where shapes matter.  Three phases:

1. **growing context** -- tokens arrive one by one, 1 to 16.  Attention's
   score matrix gains a row per token (``n × n`` voxels, the causal half);
   the recurrence gains one step at the end of its chain; the SSM's
   parallel scan gains a combine level only when ``n`` passes a power of
   two (``⌈log₂ n⌉`` levels).  Under each, a bar: the work the *new* token
   costs -- attention ``n`` scores against every key, the recurrence 1
   state update, the SSM 1 state update at inference,
2. **the whole sequence** -- what cannot be parallelised: attention computes
   all ``n²`` scores at once (depth 1), the recurrence needs ``n`` steps one
   after another, the scan ``⌈log₂ n⌉`` levels (16 → 4),
3. **memory at inference** -- GPT-2 small's KV cache grows by one column per
   token: ``12 layers × 2 (K, V) × 12 heads × 64 = 18,432`` numbers (its
   real shapes, :mod:`voxanim.gpt2`), against the fixed state of a
   Mamba of the same size (Mamba-130M: 24 layers × d_inner 1536 × d_state
   16 = 589,824 numbers, plus 24 × 1536 × 3 = 110,592 for its convolution
   window).  Each voxel is 768 numbers in both panels; the cache passes the
   state at 32 tokens and the sequence runs on to 64.
"""

from __future__ import annotations

import itertools

import numpy as np

from . import color as C
from . import layout
from .conv import Tempo
from .easing import ease_in_out, ease_out, remap, smoothstep
from .gpt2 import D_HEAD, N_HEAD, N_LAYER
from .mamba import scan_levels
from .netanim import OFF, NetAnimation, particle
from .text import TIMES

N_TOKENS = 16                 # the context of the first two phases
N_MEMORY = 64                 # tokens of the memory phase
# Mamba-130M (Gu & Dao 2023, the model matched to GPT-2 small's 124M parameters)
MAMBA_LAYERS, MAMBA_D_INNER, MAMBA_D_STATE, MAMBA_D_CONV = 24, 1536, 16, 4
KV_PER_TOKEN = N_LAYER * 2 * N_HEAD * D_HEAD                   # 18,432 numbers
MAMBA_STATE = MAMBA_LAYERS * MAMBA_D_INNER * MAMBA_D_STATE     # 589,824 numbers
MAMBA_CONV = MAMBA_LAYERS * MAMBA_D_INNER * (MAMBA_D_CONV - 1)  # 110,592 numbers
CELL = N_HEAD * D_HEAD        # numbers per voxel in the memory panels: 768
KV_ROWS = KV_PER_TOKEN // CELL                # 24 voxels per token
STATE_COLS = MAMBA_STATE // CELL // KV_ROWS   # the state as a 24 x 32 block

ATTN_AT, RNN_AT, SSM_AT = (0.19, 0.42), (0.5, 0.37), (0.81, 0.37)   # the three panels
BAR_Y = 0.84                                  # bars' baseline, layout fraction
BAR_SCALE = 0.6                               # voxel size of the bars (two voxels wide)
PPU = 11.0
KV_AT, STATE_AT = (0.29, 0.58), (0.74, 0.58)  # the memory panels
MEM_SCALE = 0.55
LEVEL_LIFT = 1.1
ATTN_INK, RNN_INK, SSM_INK = C.rgb("#d84e4e"), C.rgb("#a1814b"), C.rgb("#6370eb")
CACHE_INK, STATE_INK = C.rgb("#d84e4e"), C.rgb("#8952eb")
FRAME = (0.96, 0.96, 0.96)


def thousands(n):
    """``18,432`` style."""
    return f"{int(n):,}"


class ContextCost(NetAnimation):
    """Attention, a recurrence and an SSM on a growing sequence: work and memory."""

    PHASES = ("growing context", "whole sequence", "memory")

    def __init__(self):
        super().__init__("The cost of context", "", PPU)
        self.hold = 2.5
        n = N_TOKENS
        self.attn = self.sheet((n, n), ATTN_AT, 1.0, "attn")
        self.rnn = self.sheet((1, n), RNN_AT, 1.0, "rnn")
        self.ssm = self.sheet((1, n), SSM_AT, 1.0, "ssm")
        self.bars = {}
        for key, at in (("attn", ATTN_AT), ("rnn", RNN_AT), ("ssm", SSM_AT)):
            self.bars[key] = self.sheet((n, 2), (at[0], BAR_Y), BAR_SCALE, key + "_bar")
            b = self.bars[key].block
            b.origin = self.cam.frac(at[0], BAR_Y) - np.array([b.extent[0] / 2, b.extent[1] / 2, 0])
        self.kv = self.sheet((KV_ROWS, N_MEMORY), KV_AT, MEM_SCALE, "kv")
        self.state = self.sheet((KV_ROWS, STATE_COLS), STATE_AT, MEM_SCALE, "state")

        Tm, t = {}, 0.0

        def add(name, d, gap=0.3):
            nonlocal t
            Tm[name] = (t, t + d)
            t += d + gap

        tok = Tempo(slow=1.0, fast=0.55, ramp_in=3.0, ramp_out=2.0).durations(n)
        self.t_starts, self.t_durs = t + np.concatenate([[0.0], np.cumsum(tok)[:-1]]), tok
        add("grow", float(tok.sum()), 1.5)
        add("whole", 9.0, 1.0)
        mem = Tempo(slow=0.6, fast=0.18, ramp_in=3.0, ramp_out=2.0).durations(N_MEMORY)
        self.m_starts, self.m_durs = t + np.concatenate([[0.0], np.cumsum(mem)[:-1]]), mem
        add("memory", float(mem.sum()), 0.3)
        self.Tm = Tm
        self.finish(t)
        self.marks = [0.0, Tm["whole"][0], Tm["memory"][0] - 1.0]

    def step_of(self, t):
        """Index into :attr:`PHASES` of the phase running at time ``t``."""
        return int(np.searchsorted(self.marks, t, side="right") - 1)

    def _arrivals(self, t, name, starts, durs, n):
        """``(tokens arrived, the arriving one, its progress)``; -1 before, ``n`` after."""
        if t < self.Tm[name][0]:
            return 0, -1, 0.0
        if t >= self.Tm[name][1]:
            return n, n, 0.0
        k = int(np.searchsorted(starts, t, side="right") - 1)
        return k, k, (t - starts[k]) / durs[k]

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at time ``t`` (``reset``: 0..1 through the loop's rewind)."""
        ph = 0 if reset > 0 else self.step_of(t)
        back = 1.0 - ease_in_out(remap(reset, 0.0, 0.7))
        done, nk, uk = self._arrivals(t, "grow", self.t_starts, self.t_durs, N_TOKENS)
        n = min(N_TOKENS, done + (1 if nk < N_TOKENS and uk > 0.3 else 0))   # tokens in context
        whole = self.progress("whole", t)
        # the first panels leave before the memory panels arrive (no half-opaque overlap)
        t_mem = self.Tm["memory"][0]
        first_out = smoothstep(t_mem - 1.3, t_mem - 0.7, t)
        mem_in = smoothstep(t_mem - 0.6, t_mem, t)
        m_done, mk, um = self._arrivals(t, "memory", self.m_starts, self.m_durs, N_MEMORY)
        m = min(N_MEMORY, m_done + (1 if mk < N_MEMORY and um > 0.3 else 0))
        levels = len(scan_levels(max(n, 1)))
        self.subtitle = [
            "token %d arrives: attention scores it against all %d keys, the recurrence and the "
            "SSM update one state" % (n, n) if n > 0 else "a sequence grows one token at a time",
            "the whole sequence: attention's %d scores at once, the recurrence %d steps in turn, "
            "the scan %d levels" % (N_TOKENS ** 2, N_TOKENS, levels),
            "%d tokens: GPT-2 small's KV cache holds %s numbers, Mamba-130M's state %s at any "
            "length" % (m, thousands(KV_PER_TOKEN * m), thousands(MAMBA_STATE)),
        ][ph]
        first = (1.0 - first_out) * back
        self._attention(sc, n, nk, uk, first)
        self._recurrence(sc, n, nk, uk, whole, first)
        self._scan(sc, n, nk, uk, whole, first)
        self._bars(sc, n, whole, first)
        self._memory(sc, m, mk, um, mem_in * back)
        self.step_indicator(sc, ph, self.PHASES)

    # -- the three panels ----------------------------------------------------------------
    def _attention(self, sc, n, nk, uk, op):
        """The causal score matrix: row ``k`` has ``k + 1`` cells, one per key."""
        s = self.attn
        N = N_TOKENS
        r, c = np.divmod(np.arange(N * N), N)
        present = (r < n) & (c <= r)
        cols = np.where(present[:, None], ATTN_INK[None], OFF[None])
        s.fill(cols.astype(np.float32))
        s.show(present)
        s.fade(op)
        s.visible = op > 0.01
        if 0 <= nk < N and uk < 1:                       # the new row lights up as it is scored
            grow = ease_out(remap(uk, 0.3, 0.9))
            s.block.white[nk, :, 0] = np.where(np.arange(N) <= nk, 0.6 * (1 - grow), 0.0)
            if uk >= 0.3:
                for j in range(nk + 1):
                    s.block.present[nk, j, 0] = j <= grow * (nk + 1)
        s.cage = 0.6 * op
        if op > 0.01:
            self.caption(sc, s, "attention: n × n scores", op, above=True, offset=0.012)
            self.caption(sc, s, "%d × %d = %d" % (n, n, n * n) if n else "", op, size=0.027)

    def _recurrence(self, sc, n, nk, uk, whole, op):
        """The chain of state updates, one per token; a value runs along it in phase 2."""
        s = self.rnn
        present = np.arange(N_TOKENS) < n
        s.fill(np.where(present[:, None], RNN_INK[None], OFF[None]).astype(np.float32))
        s.show(present)
        s.fade(op)
        s.visible = op > 0.01
        if 0 <= nk < N_TOKENS and uk < 1:
            s.block.white[0, nk, 0] = 0.6 * (1 - smoothstep(0.6, 1.0, uk))
        s.cage = 0.6 * op
        if op <= 0.01:
            return
        self.caption(sc, s, "recurrence: n steps", op, above=True, offset=0.012)
        self.caption(sc, s, "%d steps, one after another" % n if n else "", op, size=0.027)
        if whole > 0:
            # one value walks the chain: the n steps happen in turn
            u = (whole * 2.5) % 1.0
            k = min(n - 1, int(u * n))
            particle(sc, s.cell(max(k - 1, 0)), s.cell(k), (u * n) % 1.0 if k > 0 else 0.5,
                     C.lighten(RNN_INK, 0.4), 0.55, lift=0.8)
            sc.probe(("rnn",), (u * n * 8, 0, 0), tol=8.0)

    def _scan(self, sc, n, nk, uk, whole, op):
        """The scan's states with its combine levels drawn as arcs above."""
        s = self.ssm
        present = np.arange(N_TOKENS) < n
        s.fill(np.where(present[:, None], SSM_INK[None], OFF[None]).astype(np.float32))
        s.show(present)
        s.fade(op)
        s.visible = op > 0.01
        if 0 <= nk < N_TOKENS and uk < 1:
            s.block.white[0, nk, 0] = 0.6 * (1 - smoothstep(0.6, 1.0, uk))
        s.cage = 0.6 * op
        if op <= 0.01 or n == 0:
            return
        levels = scan_levels(n)
        top = s.block.box((0, 1), (0, 1), (0, 1))[1][2]
        base = s.cell(0)
        step = s.cell(1) - base
        lit = int(whole * 2.5 * (len(levels) + 1)) % (len(levels) + 1) if whole > 0 else -1
        for k, off in enumerate(levels):
            z = top + LEVEL_LIFT * (k + 1)
            ink = C.lighten(SSM_INK, 0.2) if k == lit else (0.5, 0.52, 0.56)
            for t_ in range(off, n):
                a, b = base + step * (t_ - off), base + step * t_
                a[2] = b[2] = z
                self._arc(sc, a, b, ink, op * (1.0 if k == lit or lit < 0 else 0.5))
        lo, hi = s.bbox()
        hi = hi + np.array([0.0, 0.0, LEVEL_LIFT * (len(levels) + 0.6)])
        self.caption(sc, (lo, hi), "SSM: a parallel scan", op, above=True, offset=0.012)
        self.caption(sc, s, "%d level%s (log₂ %d)" % (len(levels), "s" if len(levels) != 1 else "",
                                                     n), op, size=0.027)

    def _arc(self, sc, a, b, ink, op, segments=6):
        """A low arc from ``a`` to ``b``."""
        if op <= 0.01:
            return
        lift = 0.4 * LEVEL_LIFT
        pts = [a + (b - a) * u + np.array([0.0, 0.0, lift * 4 * u * (1 - u)])
               for u in np.linspace(0, 1, segments + 1)]
        for p, q in itertools.pairwise(pts):
            sc.line(p, q, width=1.0, color=ink, opacity=op, bias=0.3)

    # -- the bars ------------------------------------------------------------------------
    def _bars(self, sc, n, whole, op):
        """Work per new token (phase 1), then sequential depth for the whole sequence."""
        depth = {"attn": 1, "rnn": N_TOKENS, "ssm": len(scan_levels(N_TOKENS))}
        per_token = {"attn": n, "rnn": 1 if n else 0, "ssm": 1 if n else 0}
        mix = smoothstep(0.1, 0.4, whole)
        inks = {"attn": ATTN_INK, "rnn": RNN_INK, "ssm": SSM_INK}
        for key, s in self.bars.items():
            h = per_token[key] + (depth[key] - per_token[key]) * mix
            rows = np.arange(N_TOKENS)[::-1]                  # row 0 is the top
            present = rows < np.ceil(h - 1e-9)
            s.fill(np.tile(inks[key], (N_TOKENS * 2, 1)))
            s.show(np.repeat(present, 2))
            s.fade(op)
            s.visible = op > 0.01 and h > 0
            if op <= 0.01:
                continue
            value = depth[key] if mix > 0.5 else per_token[key]
            self.caption(sc, s, str(value), op * (1 - 4 * mix * (1 - mix)), size=0.03,
                         above=True)
        if op <= 0.01:
            return
        y = self.fr(0.5, BAR_Y + 0.045)[1]
        label = ("work for the new token: attention n = %d, recurrence 1, SSM 1" % n
                 if mix < 0.5 else
                 "steps that must wait for each other: 1, %d, log₂ %d = %d"
                 % (N_TOKENS, N_TOKENS, depth["ssm"]))
        sc.text(0.5, y, label, 0.027, "regular", layout.CAPTION_INK,
                op * abs(1 - 2 * mix), "mt")

    # -- memory --------------------------------------------------------------------------
    def _memory(self, sc, m, mk, um, op):
        """The KV cache filling column by column against the fixed state."""
        kv, st = self.kv, self.state
        kv.visible = st.visible = op > 0.01
        if op <= 0.01:
            return
        cols = np.tile(np.arange(N_MEMORY), KV_ROWS)
        present = cols < m
        kv.fill(np.where(present[:, None], CACHE_INK[None], OFF[None]).astype(np.float32))
        kv.show(present)
        kv.fade(op)
        kv.cage = 0.6 * op
        if 0 <= mk < N_MEMORY and um < 1:
            kv.block.white[:, mk, 0] = 0.6 * (1 - smoothstep(0.6, 1.0, um))
        st.fill(np.tile(STATE_INK, (KV_ROWS * STATE_COLS, 1)))
        st.fade(op)
        st.cage = 0.6 * op
        self.caption(sc, kv, "GPT-2 small: the KV cache", op, above=True, offset=0.012)
        self.caption(sc, kv, "%d tokens %s %s = %s numbers" % (m, TIMES, thousands(KV_PER_TOKEN),
                                                               thousands(KV_PER_TOKEN * m)), op,
                     size=0.027, max_width=0.4)
        self.caption(sc, kv, "per token: %d layers %s 2 %s %d heads %s %d" % (
            N_LAYER, TIMES, TIMES, N_HEAD, TIMES, D_HEAD), op, size=0.026, offset=0.04,
            max_width=0.4, ink=layout.NOTE_INK)
        self.caption(sc, st, "Mamba-130M: the state", op, above=True, offset=0.012)
        self.caption(sc, st, "%s numbers, fixed" % thousands(MAMBA_STATE), op, size=0.027)
        self.caption(sc, st, "%d layers %s %d %s %d (+ %s conv window)" % (
            MAMBA_LAYERS, TIMES, MAMBA_D_INNER, TIMES, MAMBA_D_STATE, thousands(MAMBA_CONV)),
            op, size=0.026, offset=0.04, max_width=0.4, ink=layout.NOTE_INK)
        cross = MAMBA_STATE // KV_PER_TOKEN
        note = ("one voxel = %d numbers in both; the cache passes the state at %d tokens"
                % (CELL, cross))
        sc.text(0.5, self.fr(0.5, 0.90)[1], note, 0.027, "regular", layout.CAPTION_INK,
                op * smoothstep(cross - 4, cross, m), "mt")
        if 0 <= mk < N_MEMORY:
            sc.probe(("kv",), (mk * 8 + um * 8, 0, 0), tol=8.0)
