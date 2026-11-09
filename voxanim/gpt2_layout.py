"""Where everything in the GPT-2 animation sits, in model space.

Units: one feature (one column of a d-wide matrix) is 1 model unit wide, and
a token's row is ``HT / T`` units tall, so a matrix with one row per token is
always :data:`HT` tall and this one layout serves every prompt length.  x
runs along the model (embedding, then layer after layer, then the output), z
is up, everything lies in the y = 0 plane.  Positions are a panel's top-left
corner; panels hang down from it (-z).

Each layer occupies :data:`MOD` units of x starting at :func:`layer_x`::

    z 808 .. 150   the 12 heads' cards, 4 x 3          (GRID_*, CARDS_Z)
    z 400          concat (T x 768)    MLP hidden (T x 3072, 1/4 unit wide)
    z 230          x W_O  (T x 768)    MLP out (T x 768)
    z   0  stream: in (x 0)            + attention (x 1100)   + MLP (x 2050)
"""

from __future__ import annotations

from . import gpt2 as G

HT = 96.0                    # height of every one-row-per-token matrix
GAP = 6.0                    # between neighbouring matrices of a card
HALF = 0.5                   # W_Q / LN(x) draw their 768-long side at half scale
MAX_TOKENS = 8               # rows below HT / 8 are too thin to read

# ---- one attention head's card (the "Falk scheme": A left of A.B, B above
# it).  Offsets (x, z) of each matrix's top-left corner from the card's
# bottom-left corner.
Q_AT = (0.0, HT)                                    # Q: T x 64
S_AT = (G.D_HEAD + GAP, HT)                         # scores, then A: T x T, right of Q
KT_AT = (S_AT[0], HT + GAP + G.D_HEAD)              # K^T: 64 x T, above the scores
O_AT = (S_AT[0] + HT + GAP, HT)                     # A V: T x 64, right of A
V_AT = (O_AT[0], HT + GAP + HT)                     # V: T x 64, above A V
CARD_W, CARD_H = O_AT[0] + G.D_HEAD, V_AT[1]        # 236 x 198

# ---- the 12 cards of a layer, 4 columns x 3 rows; head 0 is top left
GRID_COLS = 4
GRID_ROWS = G.N_HEAD // GRID_COLS
GRID_COL, GRID_ROW = 260.0, 230.0                   # card pitch
GRID_X = -124.0                                     # left edge, from the layer's x
CARDS_Z = 150.0                                     # bottom of the lowest row
GRID_RIGHT = GRID_X + (GRID_COLS - 1) * GRID_COL + CARD_W          # 892
GRID_TOP = CARDS_Z + (GRID_ROWS - 1) * GRID_ROW + CARD_H           # 808

# ---- one layer, x from the layer's start
MOD = 3050.0                 # one layer's width along x
SEG_IN_X, SEG_MID_X, SEG_OUT_X = 0.0, 1100.0, 2050.0   # the stream: in, + attention, + MLP
STREAM_W = G.D_MODEL         # a stream segment is 768 wide
LEAD = 52.0                  # the stream's connector reaches this far before a layer
DROP_DX = 60.0               # a branch's vertical drop sits this far into its row
MLP_FORK_X = 1950.0          # where the MLP branch leaves the stream
HEADS_OUT_Z = 500.0          # where the heads' connector to the concatenation leaves the grid
STREAM_Z, PROJ_Z, WIDE_Z = 0.0, 230.0, 400.0           # rows: stream, x W_O / MLP out, concat / MLP

# ---- before and after the layers
EMBED_X = -1050.0            # the embeddings and x0
EMB_TOK, EMB_POS = 340.0, 180.0                        # z of the two embedding tables
OUT_X = G.N_LAYER * MOD      # final LayerNorm and logits
BARS_DX, BAR_PITCH, BAR_W, BAR_H = 950.0, 230.0, 150.0, 700.0      # next-token bars (p = 1)


def layer_x(l):
    """x of layer ``l``'s start (0-based)."""
    return l * MOD


def card_origin(l, h):
    """``(x, z)`` of the bottom-left corner of layer ``l``, head ``h``'s card."""
    r, c = divmod(h, GRID_COLS)
    return layer_x(l) + GRID_X + c * GRID_COL, CARDS_Z + (GRID_ROWS - 1 - r) * GRID_ROW
