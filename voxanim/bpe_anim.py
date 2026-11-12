"""Byte-pair encoding: GPT-2's tokenizer taking a sentence apart, merge by
merge, and the embedding rows its tokens pick.

    "The hippopotamus yawned"
     -> 3 pieces (the regular expression; a word keeps its leading space)
     -> 23 bytes, each one of 256 starting symbols (a space is written Ġ)
     -> 17 merges, lowest rank first within each piece
     -> 6 tokens:  The | Ġhipp | opot | amus | Ġyawn | ed
     -> 6 ids, 6 rows of the 50,257 x 768 embedding table

Every merge is the real one from GPT-2's ``merges.txt`` (:meth:`Tokenizer.bpe_trace
<voxanim.gpt2.Tokenizer.bpe_trace>`), applied in the order the tokenizer applies
it: the pair with the lowest rank (learned earliest, so the most frequent)
goes first, and a piece is finished when no adjacent pair is in the 50,000
rules.  A common word ("The") is done in 2 merges and is one token; a rare
one (" hippopotamus") stops after 10 merges as three.

1. **split** -- the regular expression cuts the text into words, each with
   its leading space,
2. **bytes** -- each piece becomes its UTF-8 bytes,
3. **merges** -- the 17 merges, one at a time: the two parts slide together,
   the rule's rank is shown, the rules applied so far pile up at the right,
4. **tokens** -- the 6 tokens and their ids,
5. **embed** -- each id picks its row of the embedding table: a 768-wide
   strip of numbers per token,
6. **spaces** -- "four" and " four" are different tokens (and different
   rows): the space byte merged into the word.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from . import color as C
from . import gpt2 as G
from . import layout
from .conv import Tempo
from .diagram import DATA, DATA_HI, NODE, Diagram, draw_strip
from .easing import ease_in_out, remap, smoothstep
from .gpt2_panel import diverging

SENTENCE = "The hippopotamus yawned"
PAIR = ("four", " four")            # the same word with and without its leading space

# ------------------------------------------------------------------ layout (px and fractions)

ROW_AT = (0.40, 0.36)              # centre of the sentence row (layout fractions)
CH_PX = 10.0                        # width of one character in a box (the mono advance)
PAD_PX = 5.0                        # a box's padding each side of its text
GAP_PART = 5.0                      # between the parts of one piece
GAP_PIECE = 26.0                    # between pieces
BOX_H = 26.0
ID_DY = 26.0                        # a token's id, below its box
READOUT_AT = (0.40, 0.50)         # the current merge, written out
STRIPS_AT = (0.40, 0.553)          # centre x and top y of the embedding strips
STRIP_W, STRIP_H = 470.0, 22.0      # one token's 768 numbers
STRIP_LABEL_DX = 12.0               # token names left of their strips
LIST_AT = (0.705, 0.42)             # the rules applied, top-left (layout fractions)
LIST_ROW = 22.0                     # px per line
LIST_COL = 120.0                    # px between its two columns
LIST_ROWS = 9                       # lines per column
LIST_RANK_W = 50.0                  # px: the rank column
PAIR_AT = (0.19, 0.86)             # the "four" / " four" boxes: centre x, y of the first
PAIR_ROW = 38.0                     # px between the two
PAIR_STRIP_DX = 150.0               # their strips, to the right of the boxes
PAIR_STRIP_W = 230.0
BOX_FILL = (0.975, 0.975, 0.97)
BOX_EDGE = (0.62, 0.63, 0.65)
TOKEN_EDGE = (0.45, 0.47, 0.50)     # a finished token's box
HOT_FILL = C.mix(NODE, DATA_HI, 0.45)
JOIN = 0.7                          # merge progress at which the two boxes are one
BYTES_WHOLE_OUT = (0.12, 0.25)      # bytes progress (eased) over which a piece's own text goes
BYTES_CHARS_IN = (0.28, 0.42)       # ...and its characters' come (once they stand clear)
BYTES_SPACE_SWAP = 0.06             # when a leading space starts being written Ġ
SPLIT_TEXT_OUT = (0.0, 0.1)         # split progress over which the sentence's one text goes
SPLIT_PIECES_IN = (0.18, 0.32)      # ...and the pieces' own texts come
MERGE_TEXT_OUT = (0.55, 0.72)       # closing progress over which the two parts' texts go
MERGE_TEXT_IN = (0.76, 0.92)        # ...and the merged part's comes

# ------------------------------------------------------------------ timeline (seconds)

TIMELINE = dict(intro=(0.2, 1.0), split=(1.6, 3.2), bytes=(3.8, 5.8), merges=(6.6, None),
                ids=(None, None), embed=(None, None), spaces=(None, None))
MERGE_TEMPO = Tempo(slow=1.7, fast=0.6, ramp_in=4.0, ramp_out=2.0)
WORD_PAUSE = 0.5                    # between the last merge of a piece and the next piece's first
AFTER = dict(ids=(0.8, 2.6), embed=(0.6, 7.0), spaces=(0.6, 8.0))   # (gap before, length)


@dataclass
class Merge:
    """One merge of the sentence: rule ``rank`` joins parts ``i`` and ``i + 1``
    (``left`` + ``right``) of piece ``piece``, from ``t0`` to ``t1`` s."""

    piece: int
    rank: int
    i: int
    left: str
    right: str
    t0: float
    t1: float

    def u(self, t):
        """Progress 0..1 at time ``t``."""
        return remap(t, self.t0, self.t1)


@cache
def tokenizer():
    """The one :class:`~voxanim.gpt2.Tokenizer` (its files are read once)."""
    return G.Tokenizer()


@cache
def tokenisation(text=SENTENCE):
    """``(pieces, units, traces, ids)`` for ``text``: the regex pieces, each
    piece's byte-mapped string, its ``bpe_trace`` and the token ids."""
    tok = tokenizer()
    pieces = tok.pieces(text)
    units = [tok.units(p) for p in pieces]
    traces = [tok.bpe_trace(u) for u in units]
    ids = tok.encode(text)
    assert [p for _s, parts in traces for p in parts] == [tok.dec[i] for i in ids]
    return pieces, units, traces, ids


class BPEAnimation(Diagram):
    """The ``gpt2-tokenizer`` animation (see the module docstring)."""

    PHASES = ("split", "bytes", "merges", "tokens", "embed", "spaces")

    def __init__(self, text=SENTENCE):
        super().__init__("Byte-pair encoding", "")
        self.text = text
        self.pieces, self.units, self.traces, self.ids = tokenisation(text)
        self.tokens = [p for _s, parts in self.traces for p in parts]
        self.n_bytes = sum(len(u) for u in self.units)
        self.space = tokenizer().b2u[32]                 # how GPT-2 writes a space: Ġ
        self.n_rules = len(tokenizer().ranks)
        self.Tm = dict(TIMELINE)
        self.merges = self._schedule(self.Tm["merges"][0])
        t = self.merges[-1].t1
        self.Tm["merges"] = (self.Tm["merges"][0], t)
        for name in ("ids", "embed", "spaces"):
            gap, length = AFTER[name]
            self.Tm[name] = (t + gap, t + gap + length)
            t += gap + length
        self._embeddings()
        self.hold, self.reset = 3.0, 1.5
        self.finish(t + 0.6)

    def _schedule(self, t0):
        """Every merge of the sentence in the order shown (piece by piece,
        each piece's merges in the tokenizer's order), timed by :data:`MERGE_TEMPO`."""
        flat = [(k, step) for k, (steps, _parts) in enumerate(self.traces) for step in steps]
        durations = MERGE_TEMPO.durations(len(flat))
        out, t = [], t0
        for (k, (rank, i, left, right)), d in zip(flat, durations):
            if out and out[-1].piece != k:
                t += WORD_PAUSE
            out.append(Merge(k, rank, i, left, right, t, t + d))
            t += d
        return out

    def _embeddings(self):
        """The tokens' rows of the embedding table (and the pair's), coloured
        on one scale, and the pair's cosine similarity."""
        tok = tokenizer()
        E = G.weights()["wte.weight"]
        self.pair_ids = [tok.encode(w)[0] for w in PAIR]
        rows = E[self.ids + self.pair_ids]
        cols = diverging(rows)
        self.strips = cols[:len(self.ids)]
        self.pair_strips = cols[len(self.ids):]
        a, b = E[self.pair_ids[0]], E[self.pair_ids[1]]
        self.pair_cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        self.pair_units = [tok.units(w) for w in PAIR]

    # ------------------------------------------------------------------ state
    def step_of(self, t):
        """Index into :attr:`PHASES` at time ``t``."""
        for k, name in ((5, "spaces"), (4, "embed"), (3, "ids"), (2, "merges"), (1, "bytes")):
            if t >= self.Tm[name][0]:
                return k
        return 0

    def current(self, t):
        """The merge under way at ``t`` (or None) and the merges done per piece
        (a merge counts as done once its boxes have joined)."""
        done = [0] * len(self.pieces)
        cur = None
        for m in self.merges:
            u = m.u(t)
            if u >= JOIN:
                done[m.piece] += 1
            elif u > 0:
                cur = m
        return cur, done

    def parts_at(self, k, t):
        """Piece ``k``'s parts at ``t`` and, if one of its merges is closing,
        ``(i, close)``: the pair joining and how far (0..1)."""
        cur, done = self.current(t)
        steps, _final = self.traces[k]
        parts = list(self.units[k])
        for _rank, i, _l, _r in steps[:done[k]]:
            parts[i:i + 2] = [parts[i] + parts[i + 1]]
        if cur is not None and cur.piece == k:
            return parts, (cur.i, ease_in_out(remap(cur.u(t), 0.15, JOIN)))
        return parts, None

    def row(self, t):
        """The sentence row at ``t``: per piece, ``(boxes, closing)``: its boxes
        ``[(x0, x1, text, hot)]`` in px from the row's left edge and, while one
        of its merges is closing, ``(i, close)`` (see :meth:`parts_at`); and the
        row's width.  Before the bytes phase a piece is one box; its
        characters' padding and gaps grow in with that phase."""
        sp = ease_in_out(self.progress("split", t))
        by = ease_in_out(self.progress("bytes", t))
        pieces, x = [], 0.0
        for k, piece in enumerate(self.pieces):
            if k:
                x += GAP_PIECE * sp
            boxes, closing = [], None
            if by <= 0:
                w = len(piece) * CH_PX + 2 * PAD_PX * sp
                boxes.append((x, x + w, piece, 0.0))
                x += w
            else:
                parts, closing = self.parts_at(k, t)
                for j, part in enumerate(parts):
                    if j:
                        gap = GAP_PART * by
                        if closing and j == closing[0] + 1:
                            gap -= (GAP_PART + 2 * PAD_PX) * closing[1]
                        x += gap
                    w = len(part) * CH_PX + 2 * PAD_PX * by
                    hot = 1.0 if closing and j in (closing[0], closing[0] + 1) else 0.0
                    boxes.append((x, x + w, part, hot))
                    x += w
            pieces.append((boxes, closing))
        return pieces, x

    # ------------------------------------------------------------------ frame
    def draw(self, sc, t, reset):
        """Draw the frame at ``t`` (``reset`` 0..1 during the loop reset)."""
        back = 1.0 - smoothstep(0.0, 0.8, reset)
        on = smoothstep(0.0, 1.0, self.progress("intro", t)) * back
        self._row(sc, t, on)
        self._readout(sc, t, back)
        self._rules(sc, t, back)
        self._strips(sc, t, back)
        self._pair(sc, t, back)
        self.step_indicator(sc, 0 if reset > 0 else self.step_of(t), self.PHASES)
        self.subtitle = self._words(t, reset)

    def _box(self, sc, c, w, text, op, hot=0.0, edge=BOX_EDGE, frame=1.0, text_op=1.0):
        """A rounded box ``w`` px wide centred at ``c`` with mono ``text`` in it;
        ``frame`` is the box's own opacity (0: the text alone)."""
        fill = C.mix(BOX_FILL, HOT_FILL, hot)
        self.rect(sc, c, w, BOX_H, fill, op * frame, border=C.mix(edge, DATA, hot),
                  border_w=1.2, radius=5, lift=0.2)
        self.label(sc, c, text, layout.MIN_TEXT, ink=layout.STRONG_INK, opacity=op * text_op,
                   kind="mono")

    def _row(self, sc, t, on):
        """The sentence as text, then as boxes: pieces, bytes, merging parts,
        tokens.  Texts dissolve into their successors (the sentence into its
        pieces, a piece into its characters, a merging pair into one part) so
        no two pieces of text ever stand on top of each other."""
        pieces, width = self.row(t)
        c0 = self.P(*ROW_AT)
        sp = ease_in_out(self.progress("split", t))
        by = ease_in_out(self.progress("bytes", t))
        whole = 1 - smoothstep(*BYTES_WHOLE_OUT, by)         # the piece as one box / text
        chars = smoothstep(*BYTES_CHARS_IN, by)              # one box / text per character
        tokens = smoothstep(0.0, 1.0, self.progress("ids", t))
        final = t >= self.Tm["merges"][1]
        edge = C.mix(BOX_EDGE, TOKEN_EDGE, tokens) if final else BOX_EDGE
        if sp < SPLIT_PIECES_IN[1]:                          # the sentence as one line of text
            self.label(sc, c0, self.text, layout.MIN_TEXT, kind="mono", ink=layout.STRONG_INK,
                       opacity=on * (1 - smoothstep(*SPLIT_TEXT_OUT, sp)))
        for k, (boxes, closing) in enumerate(pieces):
            if whole > 0:
                x0, x1 = boxes[0][0], boxes[-1][1]
                text = self.pieces[k] if by < BYTES_SPACE_SWAP else self.units[k]
                self._box(sc, self.shift(c0, (x0 + x1) / 2 - width / 2, 0.0), x1 - x0, text,
                          on * whole, frame=sp, text_op=smoothstep(*SPLIT_PIECES_IN, sp))
            if chars <= 0:
                continue
            for j, (x0, x1, text, hot) in enumerate(boxes):
                c = self.shift(c0, (x0 + x1) / 2 - width / 2, 0.0)
                text_op = 1 - smoothstep(*MERGE_TEXT_OUT, closing[1]) if hot else 1.0
                self._box(sc, c, x1 - x0, text, on * chars, hot, edge=edge, text_op=text_op)
                if hot and j == closing[0]:                  # the merged part's text, early
                    x1b = boxes[j + 1][1]
                    self.label(sc, self.shift(c0, (x0 + x1b) / 2 - width / 2, 0.0),
                               text + boxes[j + 1][2], layout.MIN_TEXT, kind="mono",
                               ink=layout.STRONG_INK,
                               opacity=on * chars * smoothstep(*MERGE_TEXT_IN, closing[1]))
        if tokens > 0:
            for n, (x0, x1, _text, _hot) in enumerate(b for boxes, _c in pieces for b in boxes):
                c = self.shift(c0, (x0 + x1) / 2 - width / 2, ID_DY)
                k = smoothstep(0.0, 0.3, remap(tokens, 0.12 * n, 0.12 * n + 0.4))
                self.label(sc, c, "%d" % self.ids[n], layout.MIN_TEXT, opacity=on * k,
                           kind="mono", ink=layout.CAPTION_INK)

    def _readout(self, sc, t, back):
        """The merge under way, written out: ``rank 33:  Ġ + h  →  Ġh``."""
        cur, _done = self.current(t)
        if cur is None:
            return
        u = cur.u(t)
        op = smoothstep(0.0, 0.1, u) * (1 - smoothstep(0.85, 1.0, u)) * back
        n = self.merges.index(cur) + 1
        c = self.P(*READOUT_AT)
        self.label(sc, c, "merge %d of %d: rank %s" % (n, len(self.merges), format(cur.rank, ",")),
                   layout.MIN_TEXT, anchor="mm", dy=-12, opacity=op, ink=layout.NOTE_INK)
        joined = u >= JOIN
        text = "%s + %s  →  %s" % (cur.left, cur.right, cur.left + cur.right) if joined else \
            "%s + %s" % (cur.left, cur.right)
        self.label(sc, c, text, layout.CAPTION, anchor="mm", dy=12, opacity=op,
                   ink=layout.STRONG_INK, kind="mono")

    def _rules(self, sc, t, back):
        """The rules applied so far, piling up at the right in two columns: rank, pair."""
        first = self.merges[0].t0
        op = smoothstep(first - 0.4, first, t) * back
        if op <= 0.01:
            return
        c = self.P(*LIST_AT)
        self.label(sc, c, "merges.txt: %s rules, by rank" % format(self.n_rules, ","),
                   layout.MIN_TEXT, anchor="lm", opacity=op, ink=layout.NOTE_INK)
        for n, m in enumerate(self.merges):
            k = smoothstep(JOIN, 1.0, m.u(t))
            if k <= 0:
                break
            col, row = divmod(n, LIST_ROWS)
            p = self.shift(c, col * LIST_COL, LIST_ROW * (row + 1.3))
            self.label(sc, p, "%d" % m.rank, layout.MIN_TEXT, anchor="rm", dx=LIST_RANK_W,
                       opacity=op * k, kind="mono", ink=layout.CAPTION_INK)
            self.label(sc, p, "%s %s" % (m.left, m.right), layout.MIN_TEXT, anchor="lm",
                       dx=LIST_RANK_W + 10, opacity=op * k, kind="mono", ink=layout.CAPTION_INK)

    def _strips(self, sc, t, back):
        """Each token's row of the embedding table as a 768-wide strip."""
        e = self.progress("embed", t)
        if e <= 0:
            return
        cx, y0 = STRIPS_AT
        top = self.P(cx, y0)
        n = len(self.ids)
        for i in range(n):
            k = smoothstep(0.0, 0.4, remap(e, 0.1 * i, 0.1 * i + 0.5))
            if k <= 0:
                continue
            c = self.shift(top, 0.0, (i + 0.5) * STRIP_H)
            draw_strip(self, sc, c, self.strips[i], STRIP_W, STRIP_H, k * back)
            self.label(sc, c, self.tokens[i], layout.MIN_TEXT, anchor="rm",
                       dx=-(STRIP_W / 2 + STRIP_LABEL_DX), opacity=k * back, kind="mono")
        op = smoothstep(0.0, 0.3, e) * back
        self.label(sc, top, "%d numbers per token (one row of the %s × %d table)"
                   % (G.D_MODEL, format(G.N_VOCAB, ","), G.D_MODEL), layout.MIN_TEXT,
                   anchor="mb", dy=-7, opacity=op, ink=layout.NOTE_INK)

    def _pair(self, sc, t, back):
        """``four`` and ``Ġfour``: two boxes, two ids, two rows of the table."""
        s = self.progress("spaces", t)
        if s <= 0:
            return
        c0 = self.P(*PAIR_AT)
        for n, (unit, i) in enumerate(zip(self.pair_units, self.pair_ids)):
            k = smoothstep(0.0, 0.35, remap(s, 0.15 * n, 0.15 * n + 0.45)) * back
            c = self.shift(c0, 0.0, n * PAIR_ROW)
            w = len(unit) * CH_PX + 2 * PAD_PX
            self._box(sc, c, w, unit, k, edge=TOKEN_EDGE)
            self.label(sc, c, "id %d" % i, layout.MIN_TEXT, anchor="lm", dx=w / 2 + 10,
                       opacity=k, kind="mono", ink=layout.CAPTION_INK)
            cs = self.shift(c, PAIR_STRIP_DX + PAIR_STRIP_W / 2, 0.0)
            draw_strip(self, sc, cs, self.pair_strips[n], PAIR_STRIP_W, STRIP_H, k)
        k = smoothstep(0.55, 0.85, s) * back
        c = self.shift(c0, PAIR_STRIP_DX + PAIR_STRIP_W + 12, PAIR_ROW / 2)
        self.label(sc, c, "cosine %.2f" % self.pair_cos, layout.MIN_TEXT, anchor="lm",
                   opacity=k, ink=layout.STRONG_INK)

    # ------------------------------------------------------------------ words
    def _words(self, t, reset):
        """The subtitle at ``t``."""
        Tm, n_t = self.Tm, len(self.ids)
        if reset > 0 or t < Tm["split"][0]:
            return 'GPT-2 reads "%s" as tokens, not characters: byte-pair encoding' % self.text
        if t < Tm["bytes"][0]:
            return ("a regular expression first cuts the text into %d pieces; a word keeps "
                    "its leading space" % len(self.pieces))
        if t < Tm["merges"][0]:
            return ("each piece becomes its UTF-8 bytes: %d symbols out of 256 to start from; "
                    "a space (byte 32) is written Ġ" % self.n_bytes)
        cur, _done = self.current(t)
        if cur is not None:
            return self._merge_words(cur, t)
        if t < Tm["ids"][0]:
            return ("no adjacent pair left is in the rules: %d tokens. A common word is one "
                    "token, a rare one is several" % n_t)
        if t < Tm["embed"][0]:
            return ("each token is one of %s rows of the vocabulary: its id. %s"
                    % (format(G.N_VOCAB, ","),
                       ", ".join("%s = %d" % (p, i) for p, i in zip(self.tokens[:3], self.ids))))
        if t < Tm["spaces"][0]:
            return ("each id picks its row of the embedding table: %d learned numbers, "
                    "what the model knows of the token" % G.D_MODEL)
        if t < Tm["spaces"][0] + 0.55 * (Tm["spaces"][1] - Tm["spaces"][0]):
            return ('"%s" and "%s" are different tokens (ids %d and %d): the space byte merged '
                    'into the word' % (PAIR[0], PAIR[1], self.pair_ids[0], self.pair_ids[1]))
        return ("two different rows of the table, cosine similarity %.2f: to GPT-2 the space is "
                "part of the word" % self.pair_cos)

    def _merge_words(self, m, t):
        """The subtitle during merge ``m``."""
        n, piece = self.merges.index(m) + 1, self.pieces[m.piece]
        if n == 1:
            return ('"%s": of its adjacent pairs, %s + %s has the lowest rank (%s): learned '
                    'first, so merged first' % (piece, m.left, m.right, format(m.rank, ",")))
        if m.u(t) < JOIN:
            return ('merge %d of %d in "%s": the lowest-ranked pair present is %s + %s, '
                    'rank %s' % (n, len(self.merges), piece, m.left, m.right,
                                 format(m.rank, ",")))
        return ('merge %d of %d: %s + %s → %s (rank %s of %s; a high rank is a rare pair)'
                % (n, len(self.merges), m.left, m.right, m.left + m.right,
                   format(m.rank, ","), format(self.n_rules, ",")))
