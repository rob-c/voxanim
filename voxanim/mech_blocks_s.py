"""Animations 236-237, the last two of the vector-database part of "Blocks and code" (see
:mod:`voxanim.blocks`), drawn from :mod:`voxanim.study_blocks_s`.

* :class:`Ivf` (``vector-db-ivf``): the 60,000 stored vectors cut by k-means into 64 lists,
  drawn as 64 bars of their real sizes (89 to 4,761), which stay at the top all through;
  a test digit's dot product with each of the 64 centres, the nearest 4 lit; the nearest 1,
  2 and 4 lists visited, the vectors in them counted as scanned and the true 10 nearest
  found counted; then the measured curve over all 10,000 queries, recall@10 against the
  share scanned for nprobe 1 to 64, nprobe 4 marked (99.5% found, 8.1% scanned).
* :class:`Versions` (``vector-db-versions``): test digit 0, a 7, through the first encoder
  and the retrained one, two strips of 32 numbers that differ (dot product 0.12); the
  retrained query against the old store finds five 3s; over all 10,000 queries the mixed
  search is right 7.6% of the time, below chance (10.0%); with the store embedded again,
  96.6%, and the same query finds 7s.

The picture is on the left; the code, the loop map and the evidence tag are the series'
furniture (:class:`~voxanim.blocks.Blocks`). Searching and indexing are not training: the
loop map lights no station there, and "forward" where a digit (or the store) is encoded.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from .blocks import (
    DATA,
    HEAD,
    INK,
    QUIET,
    SIZE,
    Blocks,
    Code,
    measured,
)
from .easing import ease_in_out, smoothstep
from .mech import AXIS, DIM, TRACK, dark

DATA_INK = dark(DATA, 0.3)


def eased(u, a, b):
    return ease_in_out(float(np.clip((u - a) / (b - a), 0.0, 1.0)))


def mix(a, b, s):
    """Colour ``a`` turned ``s`` of the way to ``b``."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    return tuple(float(v) for v in a + (b - a) * float(s))


def on(u, a, op, width=0.05):
    """``op`` from ``u = a`` on, faded in over ``width``."""
    return op * smoothstep(a, a + width, u)


class Searched(Blocks):
    """What the two share."""

    def at(self, k, u):
        """1 once phase ``k`` has reached ``u`` (fading in over a short while); 0 before
        phase ``k``."""
        a, b = self.spans[k]
        t0 = a + (b - a) * u
        return smoothstep(t0, t0 + 0.5, self._t)

    def column(self, sc, fx, base, h, w, ink, op, lift=0.3):
        """A bar standing on layout ``base``, ``h`` layout tall, ``w`` px wide."""
        if op <= 0.01:
            return
        hp = max(self.span(0, h)[1], 1.2)
        self.rect(sc, self.P(fx, base - self.frac(0, hp / 2)[1]), w, hp, ink, op, radius=0,
                  lift=lift)

    def bar(self, sc, x0, y, w, h, ink, op, lift=0.3):
        """A flat bar from layout ``x0``, ``w`` layout wide, ``h`` px tall, centred on ``y``."""
        if op <= 0.01 or w <= 0:
            return
        self.rect(sc, self.P(x0 + w / 2, y), max(self.span(w)[0], 1.2), h, ink, op, radius=0,
                  lift=lift)

    def small_digit(self, sc, img, fx, fy, tile, op, lift=0.2):
        if op > 0.01:
            self.digit_card(sc, np.asarray(img, float), fx, fy, tile, op, lift=lift)


# ====================================================================== 236
class Ivf(Searched):
    """64 lists; a query against the centres; the nearest 1, 2, 4 lists scanned; recall
    against scanned over all the queries."""

    TITLE = "An index: look in fewer places"
    PHASES = (("64 lists", 12.0), ("the centres", 13.0), ("visit a few", 16.0),
              ("all queries", 15.0))
    LAST = "Store, then search"
    NEXT = "A new encoder needs a new store"
    STATION = (None, None, None, None)
    COLS = (0.07, 0.465)            # the 64 columns span this
    BAR_W = 4.2                     # px
    LIST_TOP, LIST_BASE = 0.425, 0.545
    NEAR_LABEL, NEAR_TOP, NEAR_BASE = 0.625, 0.65, 0.735
    CARD = (0.1, 0.815)             # the query's digit card
    TEXT_X = 0.165                  # what is said beside it
    STEPS_AT = (0.08, 0.38, 0.68)   # phase 3: 1, 2, then 4 lists
    CHART = ((0.14, 0.44), (0.67, 0.83))

    def setup(self):
        m = self.meta = self.study.metadata
        self.sizes = np.asarray(m["sizes"], float)
        self.near = np.asarray(m["near"], float)
        n = len(self.sizes)
        x0, x1 = self.COLS
        self.xs = [x0 + (x1 - x0) * (j + 0.5) / n for j in range(n)]
        self.CODE = (
            Code("vdb-ivf-build", run=((0.0, 0), (0.45, 1))),
            Code("vdb-ivf-search", run=((0.0, 0),)),
            Code("vdb-ivf-search", run=((0.0, 1), (0.25, 2), (0.45, 3), (0.65, 4))),
            Code("vdb-ivf-search", run=((0.0, 0),)))
        t = m["query"]
        self.EVIDENCE = (measured("k-means on the 60,000, one run"),
                         measured("test digit %d against 64 centres" % t),
                         measured("test digit %d, against exact search" % t),
                         measured("10,000 queries, against exact search"))
        self.chart_ = self.chart(self.CHART, [0.0, 1.0], [0.78, 1.0], pad=0.0)

    def heading_for(self, k, u):
        return ("k-means cuts the store into 64 lists",
                "the query: how near is each centre?",
                "search only the nearest lists",
                "found against scanned, all 10,000 queries")[k]

    def subtitle_for(self, k):
        m = self.meta
        s = m["steps"]
        return [
            "k-means puts each of the %s vectors in the list of its nearest centre: lists of "
            "%s to %s." % (m["stored_text"], m["smallest_text"], m["biggest_text"]),
            "Test digit %d, a %d, is compared with the 64 centres only. The 4 nearest are lit."
            % (m["query"], m["query_label"]),
            "Searching 1, 2, then 4 lists compares %s, %s, then %s vectors and finds %d, %d, "
            "then %d of its 10 nearest." % (s[0]["scanned_text"], s[1]["scanned_text"],
                                            s[2]["scanned_text"], s[0]["found"], s[1]["found"],
                                            s[2]["found"]),
            "Over 10,000 queries, 4 lists find %s of the true 10 nearest, scanning %s. "
            "Scanned counts comparisons, not time." % (m["marked_recall_text"],
                                                      m["marked_scanned_text"]),
        ][k]

    # ------------------------------------------------------------------ what is lit
    def lit(self, j, k, u):
        """How lit list ``j`` is (0..1): from phase 3 on, the nearest 1, then 2, then 4
        light up one step at a time."""
        if k < 2:
            return 0.0
        for at, s in zip(self.STEPS_AT, self.meta["steps"], strict=True):
            if j in s["near"]:
                return 1.0 if k > 2 else smoothstep(at, at + 0.05, u)
        return 0.0

    def dimmed(self, k, u):
        """How far the lists not visited have turned grey (0..1)."""
        if k < 2:
            return 0.0
        return 1.0 if k > 2 else smoothstep(self.STEPS_AT[0], self.STEPS_AT[0] + 0.05, u)

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        how = self.dimmed(k, u)
        grow = eased(u, 0.06, 0.3) if k == 0 else 1.0
        x0 = self.COLS[0]
        self.text(sc, x0, 0.385, "%d lists · %s vectors" % (m["lists"], m["stored_text"]),
                  op * (self.at(0, 0.04) if k == 0 else 1.0), SIZE, "lm")
        full = self.LIST_BASE - self.LIST_TOP
        for j, (x, n) in enumerate(zip(self.xs, self.sizes)):
            h = full * n / m["biggest"] * grow
            g = how * (1.0 - self.lit(j, k, u))
            self.column(sc, x, self.LIST_BASE, h, self.BAR_W, mix(DATA, DIM, g),
                        op * (1.0 - 0.4 * g))
        b = op * (self.at(0, 0.3) if k == 0 else 1.0)
        big = self.xs[m["biggest_list"]]
        self.text(sc, big + 0.006, self.LIST_TOP + 0.01, m["biggest_text"], b, SIZE, "lm",
                  ink=DATA_INK)
        self.text(sc, x0, self.LIST_BASE + 0.03, "each bar: one list, %s to %s vectors"
                  % (m["smallest_text"], m["biggest_text"]), b, SIZE, "lm", ink=QUIET)
        # the query against the centres, and its digit (phases 2 and 3)
        keep = op * self.kept(1, 2)
        if keep > 0.01:
            self.centres(sc, k, u, keep)

    def centres(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(1, 0.06) if k == 1 else 1.0)
        grow = eased(u, 0.1, 0.32) if k == 1 else 1.0
        top4 = m["order"][:4]
        lit = smoothstep(0.42, 0.5, u) if k == 1 else 1.0
        self.text(sc, self.COLS[0], self.NEAR_LABEL, "q · each centre: how near", a, SIZE, "lm")
        full = self.NEAR_BASE - self.NEAR_TOP
        for j, (x, v) in enumerate(zip(self.xs, self.near)):
            h = full * max(v, 0.0) * grow
            g = 0.0 if j in top4 else lit
            self.column(sc, x, self.NEAR_BASE, h, self.BAR_W, mix(DATA, DIM, g),
                        a * (1.0 - 0.4 * g))
        self.small_digit(sc, m["query_image"], self.CARD[0], self.CARD[1], 1.8, a)
        self.text(sc, self.TEXT_X, self.CARD[1] - 0.04, "test digit %d, a %d: q"
                  % (m["query"], m["query_label"]), a, SIZE, "lm")

    # ------------------------------------------------------------------ 1-2
    def phase_1(self, sc, u, op):
        """The 64 lists, in ``steady``."""

    def phase_2(self, sc, u, op):
        """The query against the centres is in ``steady``; here, what is lit."""
        self.text(sc, self.TEXT_X, self.CARD[1], "the 4 nearest centres: lit", on(u, 0.5, op),
                  SIZE, "lm", ink=DATA_INK)

    # ------------------------------------------------------------------ 3: visit a few
    def phase_3(self, sc, u, op):
        m = self.meta
        s = m["steps"]
        a = on(u, self.STEPS_AT[0], op)
        steps = [(at, i) for at, i in zip(self.STEPS_AT, range(3), strict=True)]
        y = self.CARD[1]
        self.running(sc, self.TEXT_X, y, [(at, "%d list%s visited" % (s[i]["nprobe"],
                                                                     "" if i == 0 else "s"))
                                         for at, i in steps], u, a, size=HEAD)
        self.running(sc, self.TEXT_X, y + 0.04,
                     [(at, "%s of %s scanned · %s" % (s[i]["scanned_text"], m["stored_text"],
                                                       s[i]["share_text"]))
                      for at, i in steps], u, on(u, self.STEPS_AT[0] + 0.06, op), size=SIZE,
                     ink=DATA_INK)
        self.running(sc, self.TEXT_X, y + 0.075,
                     [(at, "found %d of its %d nearest" % (s[i]["found"], m["k"]))
                      for at, i in steps], u, on(u, self.STEPS_AT[0] + 0.12, op), size=SIZE)

    # ------------------------------------------------------------------ 4: the curve
    def phase_4(self, sc, u, op):
        m = self.meta
        ch = self.chart_
        frame = on(u, 0.04, op)
        self.text(sc, ch.x0 - 0.065, ch.y0 - 0.04, "found", frame, SIZE, "lm")
        self.axes(sc, ch, frame, xlabel="share of the store scanned",
                  xticks=[(0.0, "0%"), (0.5, "50%"), (1.0, "100%")],
                  yticks=[(0.8, "80%"), (0.9, "90%"), (1.0, "100%")])
        xs, ys = np.asarray(m["scanned"]), np.asarray(m["recall"])
        draw = eased(u, 0.12, 0.4)
        self.plot(sc, ch, xs, ys, DATA, frame, draw, dot=False)
        for x, y in zip(xs, ys, strict=True):
            if (x - xs[0]) / (xs[-1] - xs[0]) <= draw + 1e-9:
                self.disk(sc, self.P(ch.fx(x), ch.fy(y)), 3.6, DATA, frame, lift=1.4,
                          outline=0.0)
        i1 = m["nprobe"].index(1)
        b = on(u, 0.2, op)
        self.text(sc, ch.fx(xs[i1]) + 0.03, ch.fy(ys[i1]) - 0.016,
                  "1 list: %s" % m["recall_text"][i1], b, SIZE, "lm", ink=DATA_INK)
        i64 = m["nprobe"].index(64)
        c = on(u, 0.42, op)
        self.text(sc, ch.fx(xs[i64]), ch.fy(ys[i64]) - 0.035, "64 lists: %s"
                  % m["recall_text"][i64], c, SIZE, "rm", ink=DATA_INK)
        # nprobe 4, ringed, its numbers beside it
        i4 = m["nprobe"].index(m["marked"])
        d = on(u, 0.5, op)
        p = (ch.fx(xs[i4]), ch.fy(ys[i4]))
        self.disk(sc, self.P(*p), 6.5, DATA, d, ring=INK, ring_w=2.0, lift=1.45, outline=0.0)
        tx, ty = ch.fx(0.2), ch.fy(0.935)
        self.text(sc, tx, ty, "4 lists: %s found" % m["marked_recall_text"], d, HEAD, "lm",
                  strong=True)
        self.text(sc, tx, ty + 0.042, "%s scanned" % m["marked_scanned_text"], d, HEAD, "lm",
                  strong=True)
        self.arrow(sc, (tx - 0.008, ty - 0.01), (p[0] + 0.01, p[1] + 0.012), AXIS, d,
                   u=eased(u, 0.52, 0.6), width=1.6)


# ====================================================================== 237
RATES = ("same", "mixed", "chance", "reembedded")
RATE_NAMES = dict(same="one encoder", mixed="mixed encoders", chance="chance",
                  reembedded="re-embedded")


class Versions(Searched):
    """One digit, two encoders; the old store with the new query; the rate over all the
    queries; the store embedded again."""

    TITLE = "A new encoder needs a new store"
    PHASES = (("two encoders", 13.0), ("old store", 13.0), ("all queries", 14.0),
              ("embed again", 14.0))
    LAST = "An index: look in fewer places"
    NEXT = "The cost of context: attention, RNN, SSM"
    STATION = ("forward", None, None, "forward")
    CARD = (0.105, 0.5)             # the test digit
    STRIP_X = (0.215, 0.455)        # the two strips of 32
    Q_Y, Q2_Y = 0.455, 0.535
    DOT_Y = 0.595
    ROW_LABEL, ROW_Y = 0.705, 0.78   # phase 2: the five found
    FOUND_XS = (0.09, 0.155, 0.22, 0.285, 0.35)
    AGAIN_XS = (0.215, 0.268, 0.321, 0.374, 0.427)
    AGAIN_Y = 0.49
    TALLY_Y, TALLY_STEP = 0.705, 0.043
    TALLY_X, TALLY_W = 0.235, 0.15      # the bars start here; 100% is this long
    RATE_AT: ClassVar[dict] = dict(same=(2, 0.08), mixed=(2, 0.3), chance=(2, 0.5),
                                   reembedded=(3, 0.65))

    def setup(self):
        m = self.meta = self.study.metadata
        self.lo, self.hi = -0.4, 0.4
        self.CODE = (
            Code("vdb-versions", run=((0.0, 0), (0.2, 2), (0.4, 3))),
            Code("vdb-versions", run=((0.0, 4), (0.15, 5))),
            Code("vdb-versions", run=((0.0, 5),)),
            Code("vdb-reembed", run=((0.0, 0), (0.08, 1), (0.18, 2), (0.3, 3), (0.38, 4))))
        self.EVIDENCE = (measured("test digit %d, two encoders" % m["query"]),
                         measured("test digit %d, our retrained run" % m["query"]),
                         measured("all %s test digits" % m["queries_text"]),
                         measured("all %s, the store re-embedded" % m["queries_text"]))

    def heading_for(self, k, u):
        return ("the same digit, two encoders",
                "old store, new query: wrong digits",
                "a vector means something only to its encoder",
                "embed the whole store again")[k]

    def subtitle_for(self, k):
        m = self.meta
        r = m["rate_text"]
        return [
            "The same recipe trained again gives this %d a different vector of %d numbers: "
            "q · q2 is only %s." % (m["label"], m["dims"], m["dot_text"]),
            "Searched against the old store, the retrained encoder's query finds five %ds for "
            "this %d." % (m["mixed"]["labels"][0], m["label"]),
            "Over all %s test digits, mixed encoders find the right digit %s of the time: "
            "less than chance, %s." % (m["queries_text"], r["mixed"], r["chance"]),
            "Embed all %s stored digits again with the new encoder: right %s of the time."
            % (m["stored_text"], r["reembedded"]),
        ][k]

    # ------------------------------------------------------------------ what stays
    def steady(self, sc, k, u, op):
        m = self.meta
        a = op * (self.at(0, 0.04) if k == 0 else 1.0)
        self.text(sc, 0.06, 0.385, "test digit %d, a %d" % (m["query"], m["label"]), a, SIZE,
                  "lm")
        self.small_digit(sc, m["image"], self.CARD[0], self.CARD[1], 2.4, a)
        s = op * self.kept(0, 2)
        if s > 0.01:
            self.strips(sc, k, u, s)
        t = op * self.kept(2, 3)
        if t > 0.01:
            self.tally(sc, k, u, t)

    def strips(self, sc, k, u, op):
        m = self.meta
        x0, x1 = self.STRIP_X
        w = self.span(x1 - x0)[0]
        cx = (x0 + x1) / 2
        right = self.CARD[0] + 0.05
        for i, (name, y, vec, at) in enumerate((("encoder(x): q", self.Q_Y, m["q"], 0.08),
                                                ("encoder2(x): q2", self.Q2_Y, m["q2"], 0.3))):
            a = op * (self.at(0, at) if k == 0 else 1.0)
            self.text(sc, x0, y - 0.035, name, a, SIZE, "lm", ink=DATA_INK)
            self.arrow(sc, (right, self.CARD[1] + (i - 0.5) * 0.05), (x0 - 0.008, y), AXIS, a,
                       u=eased(u, at, at + 0.08) if k == 0 else 1.0, width=1.6)
            self.tiles(sc, np.atleast_2d(vec), cx, y, w, 16.0, a, lo=self.lo, hi=self.hi,
                       grid=0.25)
        b = op * (self.at(0, 0.5) if k == 0 else 1.0)
        self.text(sc, x0, self.DOT_Y, "q · q2 = %s" % m["dot_text"], b, HEAD, "lm", strong=True)
        c = op * (self.at(0, 0.62) if k == 0 else 1.0)
        self.text(sc, x0, self.DOT_Y + 0.04, "all %s digits: %s on average"
                  % (m["stored_text"], m["cross_text"]), c, SIZE, "lm", ink=QUIET)

    def tally(self, sc, k, u, op):
        m = self.meta
        self.text(sc, 0.06, self.TALLY_Y, "right digit, all %s queries" % m["queries_text"],
                  op * (self.at(2, 0.04) if k == 2 else 1.0), SIZE, "lm")
        for i, name in enumerate(RATES):
            y = self.TALLY_Y + self.TALLY_STEP * (i + 1)
            kk, uu = self.RATE_AT[name]
            shown = op * self.at(kk, uu) if k >= kk else 0.0
            if shown <= 0.01:
                continue
            self.text(sc, 0.06, y, RATE_NAMES[name], shown, SIZE, "lm")
            self.bar(sc, self.TALLY_X, y, self.TALLY_W, 12.0, TRACK, shown * 0.5, lift=0.25)
            w = self.TALLY_W * m["rate"][name]
            self.bar(sc, self.TALLY_X, y, w, 12.0, AXIS if name == "chance" else DATA, shown)
            self.text(sc, self.TALLY_X + w + 0.01, y, m["rate_text"][name], shown, SIZE, "lm",
                      ink=INK)

    # ------------------------------------------------------------------ the found digits
    def found_row(self, sc, found, xs, y, tile, u, op, start):
        """The 5 nearest stored digits, each with its label under it."""
        for i, (x, img, lab) in enumerate(zip(xs, found["images"], found["labels"],
                                              strict=True)):
            a = on(u, start + 0.04 * i, op)
            self.small_digit(sc, img, x, y, tile, a)
            self.text(sc, x, y + self.frac(0, 28 * tile / 2 + 22)[1], str(lab), a, HEAD,
                      ink=INK)

    def phase_1(self, sc, u, op):
        """The digit, the two strips and their dot product, in ``steady``."""

    def phase_2(self, sc, u, op):
        m = self.meta
        self.text(sc, 0.06, self.ROW_LABEL, "store: encoder · query: encoder2",
                  on(u, 0.06, op), SIZE, "lm", ink=DATA_INK)
        self.found_row(sc, m["mixed"], self.FOUND_XS, self.ROW_Y, 1.5, u, op, 0.2)
        self.text(sc, self.FOUND_XS[-1] + 0.042, self.ROW_Y, "not a %d" % m["label"],
                  on(u, 0.5, op), HEAD, "lm", strong=True)

    def phase_3(self, sc, u, op):
        m = self.meta
        y = self.TALLY_Y + self.TALLY_STEP * 4
        self.text(sc, 0.06, y, "mixed: %s of %s right" % (m["right_text"], m["queries_text"]),
                  on(u, 0.35, op), SIZE, "lm", ink=QUIET)
        self.text(sc, self.TALLY_X + self.TALLY_W, y, "below chance", on(u, 0.62, op), SIZE,
                  "rm", strong=True)

    def phase_4(self, sc, u, op):
        m = self.meta
        self.text(sc, self.AGAIN_XS[0] - 0.03, 0.415, "store: encoder2 · query: encoder2",
                  on(u, 0.36, op), SIZE, "lm", ink=DATA_INK)
        self.found_row(sc, m["again"], self.AGAIN_XS, self.AGAIN_Y, 1.2, u, op, 0.4)
        self.text(sc, 0.06, 0.64, "the store %s, embedded again" % m["store_shape"],
                  on(u, 0.05, op), SIZE, "lm", ink=DATA_INK)


ANIMATIONS = {"vector-db-ivf": Ivf, "vector-db-versions": Versions}
