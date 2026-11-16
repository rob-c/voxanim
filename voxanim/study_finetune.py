"""Redesigned experiments for seven of the fine-tuning animations (35-41).

The first versions (:mod:`voxanim.next_language`) could not show their concepts: every
fine-tune had a few hundred windows to memorise and a base model of 180 updates, so
instruction tuning, both merged fine-tunes and full tuning ended worse than they began on
held-out text, and the reward model saw three facts about a word's spelling and ranked
held-out pairs at chance - which left PPO, DPO and reward hacking optimising noise.

Here every fine-tune draws fresh windows of the training split at each update (no example
is seen twice), is stopped on the validation split and is reported on the test split, over
three seeds; and each task is one the base cannot already do:

* **instruction tuning**: ``Echo:`` and ``Caps:`` followed by eight real characters and a
  newline; the answer is those characters again, or in capitals. What to answer depends on
  the instruction. Answer-only against all-token loss, same minibatches.
* **reward model**: a score for (previous word, next word) from two learned 8-number word
  vectors, trained on every word pair of the training split against freshly drawn rejected
  words; beside a score per word (no context) and the first version's three spelling facts.
* **PPO, DPO, reward hacking**: the same loops as before on that reward model's scores for
  the words after 'the'. Reward hacking is run against reward models trained on 300 pairs,
  3,000 pairs and all of them, ten seeds each.
* **model merging**: one base, tuned once on the text in capitals and once on the text
  backwards; task vectors added, averaged and subtracted, and a plane of mixes.
* **prompt and prefix tuning**: the four methods adapt the base to the text in capitals.

The text in capitals or backwards is Tiny Shakespeare under an exact, stated re-encoding:
every character predicted is still one the play has. Every function without a leading
underscore that ends ``_runs`` or is called ``preferences`` is the experiment (seeded, plain
numpy, cached); the functions registered in :data:`STUDIES` wrap those numbers as four-panel
studies. :mod:`voxanim.mech_finetune` draws from both.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from functools import cache, lru_cache

import numpy as np

from . import realdata
from .cache import cached_npz
from .optim import Adam
from .study import Panel, Study

# Bump a study's number when its experiment changes: the cache is keyed on it.
VERSIONS = dict(base=1, instruction=3, preferences=2, merging=1, peft=1)
SEEDS = (0, 1, 2)
WIDTH = 32
BASE_STEPS = 1500


def _cached(name, build, **settings):
    return cached_npz("study-finetune-" + name, dict(
        version=VERSIONS[name], name=name, dataset_sha256=realdata.TEXT_SHA256,
        sources=realdata.source_hash("tinylm.py", "realdata.py", "optim.py"), **settings), build)


def _text_meta(**extra):
    return dict(
        seeds=list(SEEDS), dataset="Tiny Shakespeare / char-rnn",
        dataset_sha256=realdata.TEXT_SHA256,
        dataset_provenance={k: realdata.text_metadata(k) for k in ("train", "validation", "test")},
        **extra)


def _curve(title, caption, ys, labels, x=None, **more):
    ys = np.atleast_2d(np.asarray(ys, float))
    return Panel(title, caption, "curve", dict(
        x=np.arange(ys.shape[1]) if x is None else np.asarray(x), ys=ys, labels=list(labels),
        **more))


def _bars(title, caption, values, labels, **more):
    return Panel(title, caption, "bars", dict(values=np.asarray(values, float),
                                              labels=list(labels), **more))


def _matrix(title, caption, values, **more):
    return Panel(title, caption, "matrix", dict(values=np.asarray(values), **more))


def softmax(z):
    z = np.asarray(z, float)
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


# ---------------------------------------------------------------------- text and the base model
@cache
def _ids(split):
    return realdata.encode(realdata.text(split))


@lru_cache(maxsize=1)
def capitals():
    """Character id -> the id of its capital (itself, if it has none)."""
    vocab = realdata.vocabulary()
    table = np.arange(len(vocab))
    for i, ch in enumerate(vocab):
        if ch.islower():
            table[i] = vocab.index(ch.upper())
    return table


def config():
    from . import tinylm as lm
    return lm.Config(vocab=len(realdata.vocabulary()), width=WIDTH, ff=2 * WIDTH)


def windows(split, n, rng, length=16, dialect="plain"):
    """``n`` windows of ``length + 1`` characters from one split: as the play has them, in
    capitals, or backwards."""
    ids = _ids(split)
    seq = ids[rng.integers(0, len(ids) - length, n)[:, None] + np.arange(length + 1)]
    if dialect == "capitals":
        seq = capitals()[seq]
    elif dialect == "backwards":
        seq = seq[:, ::-1]
    return seq


@cache
def base_model(seed):
    """The base every fine-tune here starts from: a one-layer transformer trained for
    ``BASE_STEPS`` updates on fresh 21-character windows of the training split."""
    from . import tinylm as lm

    def build():
        c = config()
        P, rng = lm.init(c, seed), np.random.default_rng(1000 + seed)
        opt, loss = Adam(P, 0.003), []
        for _ in range(BASE_STEPS):
            seq = windows("train", 32, rng, 21)
            value, g = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], c)
            opt.step(P, g)
            loss.append(value)
        return dict({"P_" + k: v for k, v in P.items()}, loss=np.asarray(loss))
    a = _cached("base", build, seed=seed, steps=BASE_STEPS, width=WIDTH)
    return {k[2:]: v for k, v in a.items() if k.startswith("P_")}


def _copy(P):
    return {k: v.copy() for k, v in P.items()}


def _spread(each):
    """Mean, lowest and highest over the first axis (the seeds)."""
    each = np.asarray(each, float)
    return each.mean(0), each.min(0), each.max(0)


# ---------------------------------------------------------------------- 35 instruction tuning
INSTRUCTIONS = ("Echo:", "Caps:")
LINE = 8                                # characters of the play in an example, and in its answer
TUNE_STEPS, TUNE_EVERY = 3000, 100


def instruction_examples(split, n, rng, lines=None, task=None):
    """``(x, y, answer mask, task)``: an instruction, eight real characters, a newline and
    the answer (the characters again for ``Echo:``, their capitals for ``Caps:``)."""
    ids = _ids(split)
    if lines is None:
        lines = ids[rng.integers(0, len(ids) - LINE + 1, n)[:, None] + np.arange(LINE)]
    if task is None:
        task = rng.integers(0, 2, len(lines))
    answer = np.where(task[:, None] == 1, capitals()[lines], lines)
    head = np.stack([realdata.encode(s) for s in INSTRUCTIONS])[task]
    seq = np.c_[head, lines, np.full((len(lines), 1), realdata.encode("\n")[0]), answer]
    mask = np.zeros((len(lines), seq.shape[1] - 1))
    mask[:, -LINE:] = 1
    return seq[:, :-1], seq[:, 1:], mask, task


def answer(P, prompt, c, n=LINE):
    """What the model writes after ``prompt``: its likeliest character, ``n`` times."""
    from . import tinylm as lm
    seq = np.asarray(prompt)
    for _ in range(n):
        seq = np.c_[seq, lm.forward(P, seq, c)[:, -1].argmax(-1)]
    return seq[:, -n:]


@lru_cache(maxsize=1)
def instruction_runs():
    """Both fine-tunes on each seed's base, same minibatches, measured every
    ``TUNE_EVERY`` updates.

    ``trace[seed, run, metric, checkpoint]``: run 0 answer-only loss, run 1 all-token;
    metrics: validation answer loss, test answer loss, test answer-character accuracy, test
    loss on the eight characters of the play (which only the all-token loss grades).
    ``stop[seed, run]``: the checkpoint with the lowest validation loss; ``written[seed,
    run]`` and ``base_written[seed]``: share of test answers written out whole and exactly,
    character by character with no help, by that checkpoint and by the base. ``base[seed,
    metric]``. For seed 0, six test examples (three lines under each instruction):
    ``show_x``, ``show_y``, ``guess[run, checkpoint, example]``, ``base_guess``."""
    from . import tinylm as lm

    def build():
        c = config()
        xv, yv, mv, _ = instruction_examples("validation", 512, np.random.default_rng(501))
        xt, yt, mt, _ = instruction_examples("test", 512, np.random.default_rng(502))
        ids = _ids("test")
        # shown on screen: the first three of forty test windows that sit inside one line
        starts = np.random.default_rng(503).integers(0, len(ids) - LINE + 1, 40)
        drawn = ids[starts[:, None] + np.arange(LINE)]
        inside = (drawn != realdata.encode("\n")[0]).all(1)
        lines = np.repeat(drawn[inside][:3], 2, 0)
        xs, ys, _m, _t = instruction_examples("test", 6, None, lines, np.tile([0, 1], 3))
        cut = xt.shape[1] - LINE + 1                   # the prompt: all but the answer

        def measure(P):
            z = lm.forward(P, xt, c)
            z = z - z.max(-1, keepdims=True)
            nll = -(z - np.log(np.exp(z).sum(-1, keepdims=True)))[
                np.arange(len(yt))[:, None], np.arange(yt.shape[1]), yt]
            return [lm.loss(P, xv, yv, c, weights=mv), nll[:, -LINE:].mean(),
                    (z[:, -LINE:].argmax(-1) == yt[:, -LINE:]).mean(), nll[:, 4:4 + LINE].mean()]

        def written(P):
            return (answer(P, xt[:, :cut], c) == yt[:, -LINE:]).all(1).mean()
        trace, stop, whole, base, base_whole, guess, base_guess = [], [], [], [], [], [], None
        for seed in SEEDS:
            start = base_model(seed)
            base.append(measure(start))
            base_whole.append(written(start))
            if seed == SEEDS[0]:
                base_guess = answer(start, xs[:, :cut], c)
            trace.append([])
            stop.append([])
            whole.append([])
            for masked in (True, False):
                P, rng = _copy(start), np.random.default_rng(2000 + seed)
                opt, rows, best = Adam(P, 0.003), [], (np.inf, 0, None)
                for step in range(TUNE_STEPS + 1):
                    if step % TUNE_EVERY == 0:
                        rows.append(measure(P))
                        if rows[-1][0] < best[0]:
                            best = (rows[-1][0], len(rows) - 1, _copy(P))
                        if seed == SEEDS[0]:
                            guess.append(answer(P, xs[:, :cut], c))
                    if step == TUNE_STEPS:
                        break
                    x, y, m, _ = instruction_examples("train", 32, rng)
                    _, g = lm.loss_grads(P, x, y, c, weights=m if masked else None)
                    opt.step(P, g)
                trace[-1].append(np.asarray(rows).T)
                stop[-1].append(best[1])
                whole[-1].append(written(best[2]))
        return dict(trace=np.asarray(trace), stop=np.asarray(stop), written=np.asarray(whole),
                    base=np.asarray(base), base_written=np.asarray(base_whole),
                    guess=np.asarray(guess).reshape(2, -1, 6, LINE), base_guess=base_guess,
                    show_x=xs, show_y=ys, example_x=xt[:6], example_y=yt[:6], mask=mt[:6],
                    counts=np.asarray([TUNE_STEPS * 32, len(xv), len(xt)]))
    return _cached("instruction", build, seeds=SEEDS, steps=TUNE_STEPS, every=TUNE_EVERY,
                   base=BASE_STEPS, width=WIDTH)


def reached(trace, level=0.9):
    """The first measured update at which test answer accuracy is ``level`` or more, for
    ``trace[..., metric, checkpoint]`` (the last update + one step if it never is)."""
    acc = np.asarray(trace)[..., 2, :]
    first = np.where((acc >= level).any(-1), (acc >= level).argmax(-1), acc.shape[-1])
    return first * TUNE_EVERY


def instruction_tuning():
    r = instruction_runs()
    trace, steps = r["trace"], np.arange(r["trace"].shape[-1]) * TUNE_EVERY
    names = ["answer-only loss", "all-token loss"]
    at = reached(trace)
    final = np.take_along_axis(trace, r["stop"][:, :, None, None], 3)[..., 0]
    loss, lo, hi = _spread(trace[:, :, 1])
    acc, alo, ahi = _spread(trace[:, :, 2])
    return Study(
        "Instruction tuning on two instructions",
        "Tiny Shakespeare · Echo: and Caps: on real lines · three seeds",
        [
            _matrix("Six test examples",
                    "An instruction, eight characters of the play, a newline and the answer.",
                    np.c_[r["example_x"], r["example_y"][:, -1:]]),
            _matrix("Answer-only supervision mask",
                    "The mask grades the eight answer characters, not the instruction or the line.",
                    r["mask"]),
            _curve("Test answer loss",
                   "Mean of three seeds; both fine-tunes of a seed see the same minibatches.",
                   loss, names, steps, low=lo, high=hi, each=trace[:, :, 1],
                   base=r["base"][:, 1]),
            _curve("Test answer-character accuracy",
                   "Each fine-tune is stopped where its validation loss is lowest.",
                   acc, names, steps, low=alo, high=ahi, each=trace[:, :, 2],
                   base=r["base"][:, 2]),
        ],
        dict(
            source="Collected TinyShakespeare text; answers computed from it by the instruction",
            settings=dict(base_steps=BASE_STEPS, adaptation_steps=TUNE_STEPS, batch=32,
                          measured_every=TUNE_EVERY, answer_characters=LINE, width=WIDTH,
                          instructions=list(INSTRUCTIONS), learning_rate=0.003,
                          distinct_training_examples=int(r["counts"][0]),
                          validation_examples=int(r["counts"][1]),
                          test_examples=int(r["counts"][2])),
            results=dict(
                updates_to_90_percent=dict(zip(names, at.T.tolist())),
                stopped_at_update=(r["stop"] * TUNE_EVERY).tolist(),
                test_accuracy_at_stop=final[:, :, 2].tolist(),
                answers_written_whole=r["written"].tolist(),
                base_accuracy=r["base"][:, 2].tolist(),
                base_answers_written_whole=r["base_written"].tolist(),
                test_line_loss_at_stop=final[:, :, 3].tolist(),
                base_line_loss=r["base"][:, 3].tolist()),
            designs_tried=[
                "first version: 'Next:' + 8 characters, answer = the next 4, 256 examples, 400 "
                "updates on a 180-update base: both fine-tunes overfit, mask 18% v 21%",
                "this one, run once as a one-seed pilot (width 32, 1000-update base, 2500 "
                "updates) to see that one layer can learn the task; there the all-token run "
                "reached 90% by update 1250",
                "three seeds at 2000 updates on 1500-update bases: no all-token run reached "
                "90% (31-42% at the end), so the budget was raised to 3000 updates to see "
                "whether it ever does; nothing else changed"],
            **_text_meta(evaluation="test split; checkpoint chosen on the validation split"),
        ))


# ---------------------------------------------------------------------- 36-39 preferences
PROMPT = "the"
CANDIDATES = 48
DIM = 8
REWARD_STEPS, REWARD_EVERY = 300, 10
KINDS = ("context", "word", "spelling")
BETAS = (0.0, 0.3, 1.0)
PPO_STEPS, DPO_STEPS, DPO_BETA = 180, 180, 1.0
PUSH_STEPS, PUSH = 240, 8.0
SIZES = (300, 3000, 0)                  # training pairs for reward hacking; 0: all of them
PUSH_SEEDS = tuple(range(10))


def _words(split):
    return re.findall(r"[a-z]+", realdata.text(split).lower())


@lru_cache(maxsize=1)
def _lexicon():
    """The words scored: the 48 candidates after 'the' (as the first version chose them)
    and the 256 commonest words of the training split."""
    train = _words("train")
    after = Counter(b for a, b in itertools.pairwise(train) if a == PROMPT)
    cands = [PROMPT] + [w for w, _ in after.most_common() if w != PROMPT][:CANDIDATES - 1]
    freq = Counter(train)
    vocab = list(dict.fromkeys(cands + [w for w, _ in freq.most_common(256)]))
    index = {w: i for i, w in enumerate(vocab)}
    pairs = {}
    for split in ("train", "validation", "test"):
        b = np.asarray([(index[a], index[b]) for a, b in itertools.pairwise(_words(split))
                        if a in index and b in index])
        # after 'the' only the candidates count, so a policy over them has every pair it needs
        pairs[split] = b[(b[:, 0] != 0) | (b[:, 1] < CANDIDATES)]
    chars = Counter("".join(train))
    raw = np.asarray([[len(w), sum(ch in "aeiou" for ch in w) / len(w),
                       np.mean([np.log1p(chars[ch]) for ch in w])] for w in vocab])
    spelling = (raw - raw.mean(0)) / raw.std(0)
    ref = np.asarray([freq[w] + 1 for w in cands], float)
    held = pairs["test"][pairs["test"][:, 0] == 0][:, 1]
    held_prob = (np.bincount(held, minlength=CANDIDATES) + 0.5) / (
        len(held) + 0.5 * CANDIDATES)
    return dict(vocab=vocab, pairs=pairs, spelling=spelling, ref=ref / ref.sum(),
                true=np.log(held_prob), held=len(held))


def rejected(pairs, rng):
    """A rejected word for each pair: any other word, or after 'the' any other candidate."""
    size = np.where(pairs[:, 0] == 0, CANDIDATES, len(_lexicon()["vocab"])) - 1
    lose = (rng.random(len(pairs)) * size).astype(int)
    return lose + (lose >= pairs[:, 1])


def reward_table(kind, P):
    """``score[previous word, next word]`` of a reward model."""
    lex = _lexicon()
    n = len(lex["vocab"])
    if kind == "spelling":
        return np.tile(lex["spelling"] @ P["f"], (n, 1))
    if kind == "word":
        return np.tile(P["b"], (n, 1))
    return P["u"] @ P["v"].T + P["b"]


def _fit_reward(kind, train, rng, held):
    """One reward model: Bradley-Terry loss on ``train`` pairs, a freshly drawn rejected
    word for every pair at every update, stopped on the validation pairs."""
    lex = _lexicon()
    n = len(lex["vocab"])
    if kind == "spelling":
        P = {"f": np.zeros(3)}
    else:
        P = {"b": np.zeros(n)}
        if kind == "context":
            P.update(u=rng.normal(0, 0.1, (n, DIM)), v=rng.normal(0, 0.1, (n, DIM)))

    def gaps(S, pairs, lose):
        return S[pairs[:, 0], pairs[:, 1]] - S[pairs[:, 0], lose]
    opt, loss, val, best = Adam(P, 0.02), [], [], (np.inf, 0, None)
    for step in range(REWARD_STEPS + 1):
        S = reward_table(kind, P)
        if step % REWARD_EVERY == 0:
            val.append(np.logaddexp(0, -gaps(S, *held["validation"])).mean())
            if val[-1] < best[0]:
                best = (val[-1], step, _copy(P))
        if step == REWARD_STEPS:
            break
        lose = rejected(train, rng)
        gap = gaps(S, train, lose)
        loss.append(np.logaddexp(0, -gap).mean())
        d = -1.0 / (1.0 + np.exp(gap)) / len(gap)
        D = (np.bincount(train[:, 0] * n + train[:, 1], d, n * n)
             - np.bincount(train[:, 0] * n + lose, d, n * n)).reshape(n, n)
        if kind == "spelling":
            G = {"f": D.sum(0) @ lex["spelling"]}
        else:
            G = {"b": D.sum(0)}
            if kind == "context":
                G.update(u=D @ P["v"], v=D.T @ P["u"])
        opt.step(P, G)
    S = reward_table(kind, best[2])
    test = gaps(S, *held["test"])
    the = held["test"][0][:, 0] == 0
    return dict(P=best[2], loss=np.asarray(loss), val=np.asarray(val), stop=best[1],
                test_loss=np.logaddexp(0, -test).mean(), accuracy=(test > 0).mean(),
                the_accuracy=(test[the] > 0).mean(), reward=S[0, :CANDIDATES])


def ppo_trace(reward, beta, steps=PPO_STEPS):
    """The first version's categorical PPO (clipped surrogate, four epochs an update, a KL
    penalty to the reference) against ``reward``. Returns ``[score, held-out score, KL]``
    after each update and the policy after each."""
    lex = _lexicon()
    ref, true = lex["ref"], lex["true"]
    z, rows, dists = np.log(ref), [], []
    for _ in range(steps):
        old = softmax(z)
        adv = reward - old @ reward
        for _epoch in range(4):
            p = softmax(z)
            ratio = p / old
            active = ~(((adv > 0) & (ratio > 1.2)) | ((adv < 0) & (ratio < 0.8)))
            raw = old * ratio * adv * active
            leash = -beta * (np.log(np.maximum(p / ref, 1e-300)) + 1)
            z = z + 0.8 * (raw - p * raw.sum() + p * (leash - p @ leash))
        p = softmax(z)
        rows.append([p @ reward, p @ true, p @ np.log(np.maximum(p / ref, 1e-300))])
        dists.append(p)
    return np.asarray(rows).T, np.asarray(dists)


def dpo_trace(winners, losers, beta=DPO_BETA, steps=DPO_STEPS):
    """DPO on the pairs themselves: a table of logits that starts at the reference.
    Returns ``[loss before the update, held-out score after, KL after]`` and the policy
    before the first update and after each."""
    lex = _lexicon()
    ref, true = lex["ref"], lex["true"]
    P = {"z": np.log(ref)}
    opt, rows, dists = Adam(P, 0.05), [], [softmax(P["z"])]
    for _ in range(steps):
        margin = beta * (P["z"][winners] - P["z"][losers] - np.log(ref[winners])
                         + np.log(ref[losers]))
        factor = -beta / (1 + np.exp(margin)) / len(winners)
        g = np.zeros_like(ref)
        np.add.at(g, winners, factor)
        np.add.at(g, losers, -factor)
        opt.step(P, {"z": g})
        p = softmax(P["z"])
        rows.append([np.logaddexp(0, -margin).mean(), p @ true,
                     p @ np.log(np.maximum(p / ref, 1e-300))])
        dists.append(p)
    return np.asarray(rows).T, np.asarray(dists)


def push_trace(reward):
    """The first version's proxy pressure: ``PUSH_STEPS`` exponentiated-gradient updates
    along ``reward`` from the reference. ``[proxy score, held-out score]`` before the first
    update and after each, and the policy at each."""
    lex = _lexicon()
    z, rows, dists = np.log(lex["ref"]), [], []
    for step in range(PUSH_STEPS + 1):  # noqa: B007
        p = softmax(z)
        rows.append([p @ reward, p @ lex["true"]])
        dists.append(p)
        z = z + PUSH / PUSH_STEPS * reward
    return np.asarray(rows).T, np.asarray(dists)


@lru_cache(maxsize=1)
def preferences():
    """The preference pairs, the reward models and everything optimised against them.

    Reward models, ``[kind, seed]`` over :data:`KINDS` and :data:`SEEDS`: ``accuracy``,
    ``the_accuracy`` (held-out pairs after 'the'), ``test_loss``, ``stop``, ``loss[kind,
    seed, update]``, ``val``, ``rewards[seed, candidate]`` (the context model's scores after
    'the'); seed 0's context model as ``u``, ``v``, ``b``.
    PPO: ``ppo[seed, beta, metric, update]``, ``ppo_p[beta, update, candidate]`` (seed 0,
    starting at the reference). DPO: ``dpo[seed, metric, update]``, ``dpo_p`` (seed 0),
    ``dpo_ppo[seed, metric, update]`` (PPO at DPO's beta), ``between[seed]`` (KL from DPO's
    final policy to PPO's). Reward hacking: ``push[size, seed, 2, update]`` and
    ``push_reward[size, seed, candidate]`` over :data:`SIZES` and :data:`PUSH_SEEDS`,
    ``push_accuracy``, ``drawn`` (``[size index, seed index]`` of the run whose held-out
    score fell furthest from its best to its last) and its policy ``push_p[update,
    candidate]``."""
    def build():
        lex = _lexicon()
        train = lex["pairs"]["train"]
        fixed = np.random.default_rng(600)
        held = {s: (lex["pairs"][s], rejected(lex["pairs"][s], fixed))
                for s in ("validation", "test")}
        fits = {}
        for kind in KINDS:
            for seed in (PUSH_SEEDS if kind == "context" else SEEDS):
                fits[kind, seed] = _fit_reward(kind, train, np.random.default_rng(
                    [KINDS.index(kind), seed]), held)
        out = {key: np.asarray([[fits[k, s][key] for s in SEEDS] for k in KINDS])
               for key in ("accuracy", "the_accuracy", "test_loss", "stop", "loss", "val")}
        rewards = np.asarray([fits["context", s]["reward"] for s in SEEDS])
        the = train[train[:, 0] == 0]
        ppo, dpo, dpo_ppo, between, losers = [], [], [], [], []
        for i, seed in enumerate(SEEDS):
            runs = [ppo_trace(rewards[i], beta) for beta in BETAS]
            ppo.append([r[0] for r in runs])
            lose = rejected(the, np.random.default_rng(700 + seed))
            losers.append(lose)
            d, dp = dpo_trace(the[:, 1], lose)
            dpo.append(d)
            versus, vp = runs[BETAS.index(DPO_BETA)]
            dpo_ppo.append(versus)
            between.append(dp[-1] @ np.log(dp[-1] / vp[-1]))
            if i == 0:
                out.update(ppo_p=np.asarray([np.r_[lex["ref"][None], r[1]] for r in runs]),
                           dpo_p=dp)
        push, push_reward, push_accuracy, push_p = [], [], [], {}
        for size in SIZES:
            push.append([])
            push_reward.append([])
            push_accuracy.append([])
            for seed in PUSH_SEEDS:
                if size:
                    rng = np.random.default_rng([9, size, seed])
                    fit = _fit_reward("context", train[rng.choice(len(train), size, False)],
                                      rng, held)
                else:
                    fit = fits["context", seed]
                values, dist = push_trace(fit["reward"])
                push[-1].append(values)
                push_reward[-1].append(fit["reward"])
                push_accuracy[-1].append([fit["accuracy"], fit["the_accuracy"]])
                push_p[size, seed] = dist
        push = np.asarray(push)
        fall = push[:, :, 1].max(2) - push[:, :, 1, -1]
        drawn = np.unravel_index(int(fall.argmax()), fall.shape)
        test = held["test"]
        test_the = test[0][:, 0] == 0
        return dict(
            out, rewards=rewards, **{k: fits["context", SEEDS[0]]["P"][k] for k in "uvb"},
            spelling_f=fits["spelling", SEEDS[0]]["P"]["f"],
            winners=the[:, 1], losers=np.asarray(losers),
            test_winners=test[0][test_the][:, 1], test_losers=test[1][test_the],
            train_pairs=train[:400], train_rejected=rejected(train[:400],
                                                             np.random.default_rng(601)),
            test_pairs=test[0], test_rejected=test[1],
            ppo=np.asarray(ppo), dpo=np.asarray(dpo), dpo_ppo=np.asarray(dpo_ppo),
            between=np.asarray(between), push=push, push_reward=np.asarray(push_reward),
            push_accuracy=np.asarray(push_accuracy), drawn=np.asarray(drawn),
            push_p=push_p[SIZES[drawn[0]], PUSH_SEEDS[drawn[1]]],
            counts=np.asarray([len(train), len(held["validation"][0]), len(test[0]),
                               len(the), int(test_the.sum()), lex["held"]]))
    a = _cached("preferences", build, seeds=SEEDS, push_seeds=PUSH_SEEDS, sizes=SIZES,
                betas=BETAS, dim=DIM, steps=[REWARD_STEPS, PPO_STEPS, DPO_STEPS, PUSH_STEPS],
                dpo_beta=DPO_BETA, push=PUSH)
    lex = _lexicon()
    return dict(a, words=lex["vocab"], candidates=lex["vocab"][:CANDIDATES], ref=lex["ref"],
                true=lex["true"], x=np.arange(CANDIDATES))


def rank_correlation(a, b):
    """Spearman's rank correlation of two score lists."""
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


PREFERENCE_RULE = ("the word the play has next is preferred to a randomly drawn word; "
                   "automatic, not human feedback")
PREFERENCE_DESIGNS = [
    "first version: a linear scorer on three spelling features of 48 words after 'the', 800 "
    "pairs: 51% on held-out pairs",
    "pilot of this design on 20,000 pairs with one fixed rejected word a pair: 85% on "
    "held-out pairs but it overfit in 40 updates; changed to every training pair with a "
    "fresh rejected word at each update before anything was optimised against it",
]


def _preference_meta(d, **settings):
    counts = [int(v) for v in d["counts"]]
    return dict(
        source="Collected TinyShakespeare automatic preferences",
        settings=dict(candidate_words=list(d["candidates"]), scored_words=len(d["words"]),
                      training_pairs=counts[0], validation_pairs=counts[1],
                      test_pairs=counts[2], training_pairs_after_the=counts[3],
                      test_pairs_after_the=counts[4], reward_dimensions=DIM, **settings),
        **_text_meta(preference_rule=PREFERENCE_RULE))


def reward_model():
    d = preferences()
    gap = np.linspace(-6, 6, 101)
    acc = d["accuracy"]
    mean, lo, hi = _spread(acc.T)
    reward = d["rewards"][0]
    return Study(
        "A reward model from preference pairs",
        "Tiny Shakespeare · the next word against a random word · three seeds",
        [
            _curve("Pairwise logistic link",
                   "This analytic curve maps a score gap to a preference probability.",
                   1 / (1 + np.exp(-gap)), ["P(preferred)"], gap),
            _curve("Reward-model training, seed 0",
                   "Pair loss of the context model on the training pairs, a fresh rejected "
                   "word for each at every update.",
                   d["loss"][0, 0], ["training pair loss"],
                   val=d["val"][0, 0], val_x=np.arange(len(d["val"][0, 0])) * REWARD_EVERY,
                   stop=d["stop"][0, 0]),
            _curve("Held-out frequency and learned score after 'the'",
                   "Candidate IDs denote real words; the held-out frequency is an evaluation "
                   "score, not human utility.",
                   [d["true"], reward], ["held-out log frequency", "learned score"], d["x"],
                   rank_correlation=np.asarray([rank_correlation(r, d["true"])
                                                for r in d["rewards"]])),
            _bars("Held-out pairs ranked the right way round",
                  "Mean of three seeds on the same test-split pairs; a coin flip is 0.5.",
                  np.r_[mean, 0.5],
                  ["two word vectors (context)", "a score per word", "three spelling facts",
                   "coin flip"], low=np.r_[lo, 0.5], high=np.r_[hi, 0.5], each=acc,
                  after_the=d["the_accuracy"]),
        ],
        dict(_preference_meta(d, steps=REWARD_STEPS, learning_rate=0.02,
                              stopped_at_update=d["stop"].tolist(), kinds=list(KINDS)),
             results=dict(heldout_pair_accuracy=dict(zip(KINDS, acc.tolist())),
                          heldout_pair_accuracy_after_the=dict(
                              zip(KINDS, d["the_accuracy"].tolist())),
                          heldout_pair_loss=dict(zip(KINDS, d["test_loss"].tolist()))),
             designs_tried=PREFERENCE_DESIGNS))


def rlhf_ppo():
    d = preferences()
    runs = d["ppo"]                                     # seed, beta, metric, update
    labels = ["beta %g" % b for b in BETAS]
    return Study(
        "PPO against the learned reward model",
        "Tiny Shakespeare word-choice bandit after 'the' · three reward-model seeds",
        [
            _curve("Reward-model score of the policy, seed 0",
                   "Categorical logits receive clipped PPO updates.",
                   runs[0, :, 0], labels, each=runs[:, :, 0]),
            _curve("Divergence from the reference, seed 0",
                   "A KL penalty holds the policy to the training text's word frequencies.",
                   runs[0, :, 2], labels, each=runs[:, :, 2]),
            _curve("Held-out corpus score, seed 0",
                   "Expected log frequency after 'the' in held-out text; not human reward.",
                   runs[0, :, 1], labels, each=runs[:, :, 1]),
            _curve("Final next-word distributions, seed 0",
                   "Candidate IDs refer to the recorded real-word vocabulary.",
                   np.r_[d["ppo_p"][:, -1], d["ref"][None]], labels + ["reference"], d["x"]),
        ],
        dict(_preference_meta(d, steps=PPO_STEPS, epochs=4, clip=0.2, betas=list(BETAS)),
             results=dict(final_score=runs[:, :, 0, -1].tolist(),
                          final_drift_nats=runs[:, :, 2, -1].tolist(),
                          final_heldout_score=runs[:, :, 1, -1].tolist()),
             designs_tried=PREFERENCE_DESIGNS + [
                 "penalties 0, 0.15 and 1 (the first version's, set for a scorer whose scores "
                 "spanned under one unit) changed to 0, 0.3 and 1 before the runs were "
                 "compared: this reward model's scores span several units"]))


def dpo():
    d = preferences()
    run, versus = d["dpo"], d["dpo_ppo"]                # seed, metric, update
    p = d["dpo_p"][-1]
    return Study(
        "DPO against PPO on the same preference pairs",
        "Tiny Shakespeare · pairs after 'the' · one table of logits each · three seeds",
        [
            _curve("DPO loss, seed 0", "The policy trains on the pairs after 'the' directly.",
                   run[0, 0], ["DPO logistic loss"], each=run[:, 0]),
            _curve("Implicit score across the candidate words, seed 0",
                   "Beta times log policy/reference, beside the reward model's score.",
                   [DPO_BETA * np.log(p / d["ref"]), d["rewards"][0]],
                   ["implicit preference score", "reward model"], d["x"]),
            _curve("Held-out corpus score, seed 0",
                   "Both policies are tables of 48 logits that start at the reference.",
                   [run[0, 1], versus[0, 1]], ["DPO corpus score", "PPO corpus score"],
                   each=np.stack([run[:, 1], versus[:, 1]], 1)),
            _curve("KL drift, seed 0", "Divergence of each policy from the reference.",
                   [run[0, 2], versus[0, 2]], ["DPO KL nats", "PPO KL nats"],
                   each=np.stack([run[:, 2], versus[:, 2]], 1), between=d["between"]),
        ],
        dict(_preference_meta(d, steps=DPO_STEPS, beta=DPO_BETA, learning_rate=0.05),
             results=dict(final_heldout_score=dict(dpo=run[:, 1, -1].tolist(),
                                                   ppo=versus[:, 1, -1].tolist()),
                          final_drift_nats=dict(dpo=run[:, 2, -1].tolist(),
                                                ppo=versus[:, 2, -1].tolist()),
                          kl_from_dpo_policy_to_ppo_policy=d["between"].tolist()),
             designs_tried=PREFERENCE_DESIGNS + [
                 "first version: a one-token transformer for DPO against a table for PPO, "
                 "beta 0.3; here both are tables and beta is 1, where DPO's optimum and "
                 "PPO's are the same policy if the reward model fits the pairs"]))


def reward_hacking():
    d = preferences()
    push = d["push"]                                    # size, seed, 2, update
    size, seed = (int(v) for v in d["drawn"])
    values = push[size, seed]
    held = push[:, :, 1]
    mean, lo, hi = _spread(np.swapaxes(held, 0, 1))
    peak = int(values[1].argmax())
    names = ["%d pairs" % s if s else "all %d pairs" % d["counts"][0] for s in SIZES]
    worse = (held[:, :, -1] < held[:, :, 0] - 0.05).sum(1)
    fell = (held.max(2) - held[:, :, -1] > 0.05).sum(1)
    return Study(
        "How hard can a reward model be optimised?",
        "Tiny Shakespeare · reward models trained on 300, 3,000 and all pairs · ten seeds each",
        [
            _curve("Proxy and held-out score of each word, the drawn reward model",
                   "Of the thirty reward models, the one whose run fell furthest from its best.",
                   [d["push_reward"][size, seed], d["true"]],
                   ["learned proxy", "held-out log frequency"], d["x"]),
            _curve("Proxy pressure on the drawn reward model",
                   "Every exponentiated-gradient update is computed; held-out text scores it.",
                   values, ["proxy expectation", "held-out expectation"],
                   np.linspace(0, PUSH, PUSH_STEPS + 1)),
            _matrix("Recorded next-word probability mass, the drawn run",
                    "Rows are every sixth update; columns are candidate word IDs.",
                    d["push_p"][::6]),
            _curve("Held-out score under pressure, by reward-model training size",
                   "Mean of ten seeds for each size; every seed is kept.",
                   mean, names, np.arange(PUSH_STEPS + 1), low=lo, high=hi, each=held,
                   proxy=push[:, :, 0], accuracy=d["push_accuracy"], ends_worse=worse,
                   falls_from_best=fell,
                   marks=np.asarray([values[1, 0], values[1, peak], values[1, -1]])),
        ],
        dict(_preference_meta(d, steps=PUSH_STEPS, pressure=PUSH, sizes=list(SIZES),
                              push_seeds=list(PUSH_SEEDS), drawn_size=SIZES[size],
                              drawn_seed=PUSH_SEEDS[seed],
                              drawn_rule="of the thirty runs, the one whose held-out score "
                              "fell furthest from its best to its last; labelled so on screen"),
             results=dict(heldout_start=float(held[0, 0, 0]),
                          heldout_best=held.max(2).tolist(), heldout_end=held[:, :, -1].tolist(),
                          best_at_update=held.argmax(2).tolist(),
                          runs_ending_below_start=dict(zip(names, worse.tolist())),
                          runs_falling_from_best=dict(zip(names, fell.tolist())),
                          proxy_rise=(push[:, :, 0, -1] - push[:, :, 0, 0]).tolist(),
                          reward_model_accuracy=d["push_accuracy"][:, :, 0].tolist()),
             designs_tried=PREFERENCE_DESIGNS + [
                 "the pressure run on the full reward model alone (a pilot): its favourite "
                 "word is the held-out text's favourite, so held-out score only rose; the "
                 "study therefore varies how good the proxy is",
                 "five seeds a size, the drawn run to be the 300-pair model with the median "
                 "fall: three of those five never fell, so the median run shows no hacking; "
                 "raised to ten seeds a size, all thirty runs shown, and the drawn run "
                 "changed to the furthest fall of the thirty, labelled as that"]))


# ---------------------------------------------------------------------- 41 model merging
TASKS = ("capitals", "backwards")
MERGE_STEPS, MERGE_EVERY, MERGE_RATE = 1000, 50, 0.001
MERGE_A = np.linspace(-0.25, 1.25, 13)      # how much of task A's vector
MERGE_B = np.linspace(-1.25, 1.25, 21)      # how much of task B's
MIXES = (("base", 0.0, 0.0), ("tuned on A", 1.0, 0.0), ("tuned on B", 0.0, 1.0),
         ("average", 0.5, 0.5), ("sum", 1.0, 1.0), ("A minus B", 1.0, -1.0),
         ("minus B", 0.0, -1.0))
MEASURED = TASKS + ("plain",)


@lru_cache(maxsize=1)
def merging_runs():
    """Two fine-tunes of each seed's base (A: the text in capitals; B: the text backwards),
    each stopped on its own validation loss, and weight arithmetic on their task vectors.

    ``scores[seed, text, mix]`` over :data:`MEASURED` and :data:`MIXES`: test loss;
    ``norms[seed, task]``, ``cosine[seed]``, ``stop[seed, task]`` (updates),
    ``val[seed, task, checkpoint]``; for seed 0 ``plane[text, i over MERGE_B, j over
    MERGE_A]``, the test loss of ``base + a * A + b * B``; ``sample[task, row]``: training
    windows."""
    from . import tinylm as lm

    def build():
        c = config()
        val = {t: windows("validation", 384, np.random.default_rng(801), dialect=t)
               for t in MEASURED}
        test = {t: windows("test", 384, np.random.default_rng(802), dialect=t)
                for t in MEASURED}

        def losses(P, held=test):
            return [lm.loss(P, held[t][:, :-1], held[t][:, 1:], c) for t in MEASURED]
        scores, norms, cosine, stops, vals, plane = [], [], [], [], [], None
        for seed in SEEDS:
            base, fine = base_model(seed), []
            stops.append([])
            vals.append([])
            for task in TASKS:
                P, rng = _copy(base), np.random.default_rng(3000 + seed)
                opt, best, rows = Adam(P, MERGE_RATE), (np.inf, 0, None), []
                for step in range(MERGE_STEPS + 1):
                    if step % MERGE_EVERY == 0:
                        rows.append(lm.loss(P, val[task][:, :-1], val[task][:, 1:], c))
                        if rows[-1] < best[0]:
                            best = (rows[-1], step, _copy(P))
                    if step == MERGE_STEPS:
                        break
                    seq = windows("train", 32, rng, dialect=task)
                    _, g = lm.loss_grads(P, seq[:, :-1], seq[:, 1:], c)
                    opt.step(P, g)
                fine.append(best[2])
                stops[-1].append(best[1])
                vals[-1].append(rows)
            tau = [{k: p[k] - base[k] for k in base} for p in fine]
            flat = [np.concatenate([t[k].ravel() for k in sorted(t)]) for t in tau]
            norms.append([np.linalg.norm(v) for v in flat])
            cosine.append(flat[0] @ flat[1] / norms[-1][0] / norms[-1][1])

            def mix(a, b):
                return {k: base[k] + a * tau[0][k] + b * tau[1][k] for k in base}  # noqa: B023
            scores.append(np.asarray([losses(mix(a, b)) for _name, a, b in MIXES]).T)
            if seed == SEEDS[0]:
                plane = np.asarray([[losses(mix(a, b)) for a in MERGE_A]
                                    for b in MERGE_B]).transpose(2, 0, 1)
        sample = np.stack([windows("train", 3, np.random.default_rng(803), dialect=t)
                           for t in TASKS])
        return dict(scores=np.asarray(scores), norms=np.asarray(norms),
                    cosine=np.asarray(cosine), stop=np.asarray(stops), val=np.asarray(vals),
                    plane=plane, sample=sample,
                    size=np.asarray(sum(v.size for v in base_model(SEEDS[0]).values())))
    return _cached("merging", build, seeds=SEEDS, steps=MERGE_STEPS, rate=MERGE_RATE,
                   base=BASE_STEPS, width=WIDTH, mixes=[m[0] for m in MIXES])


def model_merging():
    r = merging_runs()
    names = [m[0] for m in MIXES]
    mean, lo, hi = _spread(r["scores"])                 # text, mix
    return Study(
        "Merging two fine-tunes by weight arithmetic",
        "Tiny Shakespeare in capitals (A) and backwards (B) · three seeds",
        [
            _matrix("Training windows of the two tasks",
                    "The same play, in capitals and backwards: every character is the play's.",
                    r["sample"].reshape(-1, r["sample"].shape[-1])),
            _bars("Lengths of the two task vectors",
                  "Each task vector is the tuned weights minus the shared base weights.",
                  r["norms"].mean(0), ["task A", "task B"], each=r["norms"],
                  cosine=r["cosine"]),
            _bars("Test loss on A (capitals)",
                  "Mean of three seeds; weight arithmetic involves no further training.",
                  mean[0], names, low=lo[0], high=hi[0], each=r["scores"][:, 0]),
            _bars("Test loss on B (backwards)",
                  "Mean of three seeds on text never used for training or stopping.",
                  mean[1], names, low=lo[1], high=hi[1], each=r["scores"][:, 1],
                  plain=mean[2], plain_each=r["scores"][:, 2]),
        ],
        dict(
            source="Collected TinyShakespeare text, re-encoded in capitals and backwards",
            settings=dict(steps=MERGE_STEPS, learning_rate=MERGE_RATE, base_steps=BASE_STEPS,
                          width=WIDTH, tasks=list(TASKS), mixes=[list(m) for m in MIXES],
                          stopped_at_update=r["stop"].tolist()),
            results=dict(test_loss_on_A=dict(zip(names, r["scores"][:, 0].T.tolist())),
                         test_loss_on_B=dict(zip(names, r["scores"][:, 1].T.tolist())),
                         test_loss_on_plain_text=dict(zip(names, r["scores"][:, 2].T.tolist())),
                         cosine=r["cosine"].tolist()),
            designs_tried=[
                "first version: two halves of the same text as the two tasks, 192 windows "
                "each, 300 updates: the halves are one distribution and both fine-tunes "
                "overfit, so no merge and neither fine-tune beat the base",
                "this one piloted on two seeds at learning rate 0.003 for 600 updates and at "
                "0.001 for 1000: the same picture (average helps both, the full sum "
                "overshoots); the smaller rate kept, for the shorter task vectors"],
            **_text_meta(evaluation="test split; checkpoints chosen on the validation split"),
        ))


# ---------------------------------------------------------------------- 40 prompt and prefix tuning
METHODS = ("soft prompt", "KV prefix", "LoRA", "full tuning")
RATES = (0.003, 0.01, 0.03)
PEFT_STEPS, PEFT_EVERY, RANK = 600, 50, 2


@lru_cache(maxsize=1)
def peft_runs():
    """Four ways to adapt each seed's base to the text in capitals.

    ``trace[seed, method, 0 validation | 1 test, checkpoint]``; ``stop[seed, method]``
    (checkpoint with the lowest validation loss); ``final[seed, method]``: test loss there;
    ``base[seed]``: the untouched base's test loss; ``sizes[method]``; ``rate[method]``:
    the learning rate of :data:`RATES` with the lowest validation loss on seed 0, used for
    every seed; ``search[method, rate]``: those validation losses."""
    from . import next_language as L
    from . import tinylm as lm

    def build():
        c = config()
        heads, dh = c.heads, c.width // c.heads
        val = windows("validation", 512, np.random.default_rng(901), dialect="capitals")
        test = windows("test", 512, np.random.default_rng(902), dialect="capitals")
        sizes = {}

        def run(seed, method, rate):
            P, rng = _copy(base_model(seed)), np.random.default_rng(4000 + seed)
            start = np.random.default_rng(4100 + seed)
            extra = {}
            if method == "soft prompt":
                extra = {"prompt": start.normal(0, 0.02, (1, RANK, c.width))}
            elif method == "KV prefix":
                extra = {"prefix_k": start.normal(0, 0.02, (1, heads, RANK, dh)),
                         "prefix_v": start.normal(0, 0.02, (1, heads, RANK, dh))}
            elif method == "LoRA":
                extra = {"A": start.normal(0, 0.03, (c.width, RANK)),
                         "B": np.zeros((RANK, c.vocab))}
            trainable = P if method == "full tuning" else extra
            sizes[method] = sum(v.size for v in trainable.values())
            opt, rows = Adam(trainable, rate), []
            for step in range(PEFT_STEPS + 1):
                if step % PEFT_EVERY == 0:
                    rows.append([L.adaptation_loss(P, extra, h[:, :-1], h[:, 1:], c, method)
                                 for h in (val, test)])
                if step == PEFT_STEPS:
                    break
                seq = windows("train", 32, rng, dialect="capitals")
                _, gp, ga = L.adaptation_loss(P, extra, seq[:, :-1], seq[:, 1:], c, method, True)
                opt.step(trainable, gp if method == "full tuning" else ga)
            return np.asarray(rows).T
        search = {(m, r): run(SEEDS[0], m, r) for m in METHODS for r in RATES}
        grid = np.asarray([[search[m, r][0].min() for r in RATES] for m in METHODS])
        pick = grid.argmin(1)
        trace = np.asarray([[search[m, RATES[pick[i]]] if seed == SEEDS[0]
                             else run(seed, m, RATES[pick[i]]) for i, m in enumerate(METHODS)]
                            for seed in SEEDS])
        stop = trace[:, :, 0].argmin(-1)
        final = np.take_along_axis(trace[:, :, 1], stop[..., None], -1)[..., 0]
        base = [lm.loss(base_model(s), test[:, :-1], test[:, 1:], c) for s in SEEDS]
        return dict(trace=trace, stop=stop, final=final, base=np.asarray(base),
                    sizes=np.asarray([sizes[m] for m in METHODS]),
                    rate=np.asarray(RATES)[pick], search=grid, sample=test[:3])
    return _cached("peft", build, seeds=SEEDS, steps=PEFT_STEPS, rates=RATES, rank=RANK,
                   base=BASE_STEPS, width=WIDTH)


def gap_closed(r):
    """``[seed, method]``: how much of the way from the base's test loss to full tuning's
    each method went."""
    base = r["base"][:, None]
    return (base - r["final"]) / (base - r["final"][:, -1:])


def prompt_and_prefix_tuning():
    r = peft_runs()
    steps = np.arange(r["trace"].shape[-1]) * PEFT_EVERY
    mean, lo, hi = _spread(r["trace"][:, :, 1])
    final, flo, fhi = _spread(r["final"])
    closed = gap_closed(r)
    return Study(
        "Parameter-efficient adaptation to the text in capitals",
        "Tiny Shakespeare · base trained on the plain text · three seeds",
        [
            _bars("Trainable numbers", "Only the adapter, or the whole model, is optimised.",
                  r["sizes"], METHODS),
            _curve("Test loss on the text in capitals",
                   "Mean of three seeds; every method starts from the same base.",
                   mean, METHODS, steps, low=lo, high=hi, each=r["trace"][:, :, 1],
                   base=r["base"]),
            _bars("Test loss where validation loss was lowest",
                  "Mean of three seeds; a smaller adapter is not assumed to match full tuning.",
                  final, METHODS, low=flo, high=fhi, each=r["final"],
                  base=np.asarray(r["base"].mean()), closed=closed.mean(0),
                  closed_each=closed),
            _bars("Fraction of the model trained",
                  "Soft prompt and KV prefix learn vectors; LoRA changes the output matrix.",
                  r["sizes"] / r["sizes"][-1], METHODS),
        ],
        dict(
            source="Collected TinyShakespeare text, re-encoded in capitals",
            settings=dict(steps=PEFT_STEPS, rank=RANK, width=WIDTH, base_steps=BASE_STEPS,
                          learning_rates_tried=list(RATES),
                          learning_rate=dict(zip(METHODS, r["rate"].tolist())),
                          learning_rate_rule="lowest validation loss on seed 0, per method",
                          stopped_at_update=(r["stop"] * PEFT_EVERY).tolist()),
            results=dict(base_test_loss=r["base"].tolist(),
                         test_loss=dict(zip(METHODS, r["final"].T.tolist())),
                         gap_to_full_tuning_closed=dict(zip(METHODS, closed.T.tolist()))),
            designs_tried=[
                "first version: adapt from the first half of the text to the second, 192 "
                "windows, 220 updates: the halves are one distribution, adapters moved "
                "0.02-0.06 nats and full tuning overfit",
                "this one piloted on one seed at the three learning rates before the rule "
                "(best of three on validation, per method) was fixed"],
            **_text_meta(evaluation="test split; checkpoints chosen on the validation split"),
        ))


STUDIES = {
    "instruction-tuning": instruction_tuning,
    "reward-model": reward_model,
    "rlhf-ppo": rlhf_ppo,
    "dpo": dpo,
    "reward-hacking": reward_hacking,
    "prompt-and-prefix-tuning": prompt_and_prefix_tuning,
    "model-merging": model_merging,
}
