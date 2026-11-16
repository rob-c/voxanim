"""The next vision, generative modelling and evaluation studies.

All plots are computed from seeded runs. Modest-resolution experiments say so
on screen; numerical outcomes are never replaced by the proposal's hypothesis.
"""

from __future__ import annotations

import itertools
import json
from functools import cache, lru_cache

import numpy as np

from . import next_vision_models as vm
from . import realdata
from . import toy2d as toy
from .cache import DEFAULT_CACHE, cached_npz, sha256_of
from .optim import Adam
from .study import Panel, Study

RUN_EVIDENCE = {}
RUN_KEYS = dict(
    zip(
        [n for n in range(52, 74) if n != 57],
        [
            "vit_two_layer",
            "clip_symmetric",
            "adversarial",
            "composed_detector",
            "unet_composed_segment",
            "style",
            "gan_mnist",
            "flow_mnist",
            "flow_matching_mnist",
            "cfg_diversity",
            "ddim",
            "vae_strong_decoder",
            "vq_restart",
            "pixelcnn",
            "bias_variance_mnist",
            "cross_validation_mnist",
            "svm_mnist",
            "boosting_mnist",
            "gmm_mnist",
            "dimension_mnist",
            "tsne",
        ],
        strict=True,
    )
)

SEED = 51
VERSION = 4
NAMES = (
    "receptive-field",
    "vit-attention",
    "clip-contrastive",
    "adversarial-examples",
    "object-detection-basics",
    "semantic-segmentation",
    "batchnorm-in-cnns",
    "style-transfer",
    "gan-training",
    "normalizing-flows",
    "flow-matching",
    "classifier-free-guidance",
    "ddpm-vs-ddim",
    "vae-posterior-collapse",
    "vq-vae-codebook",
    "autoregressive-pixels",
    "bias-variance",
    "cross-validation",
    "svm-and-kernels",
    "gradient-boosting",
    "gaussian-mixture-em",
    "curse-of-dimensionality",
    "tsne-and-umap",
    "roc-and-calibration",
)


def curve(title, caption, ys, labels, x=None, notes=()):
    ys = np.atleast_2d(ys)
    return Panel(
        title,
        caption,
        "curve",
        dict(x=np.arange(ys.shape[1]) if x is None else x, ys=ys, labels=labels),
        notes,
    )


def matrix(title, caption, values, notes=()):
    return Panel(title, caption, "matrix", dict(values=values), notes)


def images(title, caption, values, labels, notes=()):
    return Panel(title, caption, "images", dict(values=np.asarray(values), labels=labels), notes)


def bars(title, caption, values, labels, notes=()):
    return Panel(title, caption, "bars", dict(values=np.asarray(values), labels=labels), notes)


def points(title, caption, xy, classes=None, notes=()):
    d = dict(xy=np.asarray(xy))
    if classes is not None:
        d["classes"] = np.asarray(classes)
    return Panel(title, caption, "points", d, notes)


def result(name, subtitle, panels, **metadata):
    if name in {
        "receptive-field",
        "batchnorm-in-cnns",
        "style-transfer",
        "ddpm-vs-ddim",
        "roc-and-calibration",
        "adversarial-examples",
    }:
        metadata["pretrained_mnist_model_seed"] = 0
    if name in {"receptive-field", "batchnorm-in-cnns", "roc-and-calibration"}:
        metadata["seed"] = None
        metadata["method"] = (
            "Analysis of a fixed NumPy-trained MNIST checkpoint; no new training run"
        )
    key = RUN_KEYS.get(51 + NAMES.index(name))
    if key in RUN_EVIDENCE:
        evidence = RUN_EVIDENCE[key]
        metadata["run_evidence"] = {
            k: evidence[k] for k in ("run_id", "artifact", "artifact_sha256", "source_sha256")
        }
        metadata["run_settings"] = evidence["settings"]
    return Study(
        name.replace("-", " ").capitalize(),
        subtitle,
        tuple(panels),
        dict(
            source="NumPy measurements on MNIST",
            dataset="MNIST",
            review_only=True,
            **{"seed": SEED, **metadata},
        ),
    )


def cached(name, build, **settings):
    source_files = ("next_vision.py", "next_vision_models.py", "toy2d.py", "mnist.py", "optim.py")
    optimization = {
        "vit_two_layer": dict(optimizer="Adam", lr=0.002, batch=32),
        "clip_symmetric": dict(optimizer="Adam", lr=0.006, batch=64, temperature=0.15),
        "adversarial": dict(
            optimizer="Adam", lr=0.0005, clean_batch=64, attack_epsilon=0.1, base_seed=0
        ),
        "composed_detector": dict(optimizer="Adam", lr=0.004, batch=96, widths=[64, 32]),
        "unet_composed_segment": dict(optimizer="Adam", lr=0.006, batch=16),
        "style": dict(optimizer="normalized pixel gradient descent", lr=0.05, pretrained_seed=0),
        "gan_mnist": dict(optimizer="Adam", generator_lr=0.002, discriminator_lr=0.001, batch=96),
        "flow_mnist": dict(optimizer="Adam", lr=0.003, batch=128),
        "flow_matching_mnist": dict(optimizer="Adam", lr=0.003, rectification_lr=0.001, batch=96),
        "cfg_diversity": dict(optimizer="Adam", lr=0.003, batch=96, classifier_lr=0.005),
        "ddim": dict(pretrained_seed=0, optimizer="none: fixed-checkpoint sampling"),
        "vae_strong_decoder": dict(optimizer="Adam", lr=0.004, batch=96),
        "vq_restart": dict(optimizer="Adam", lr=0.004, batch=96, commitment=0.25),
        "pixelcnn": dict(optimizer="Adam", lr=0.005, batch=64),
        "bias_variance_mnist": dict(
            optimizer="NumPy polynomial least squares",
            target="right-half mean ink",
            inputs="left-half mean ink",
        ),
        "cross_validation_mnist": dict(
            optimizer="kernel ridge linear solve",
            input_features="49 pooled pixels; no fitted preprocessing",
        ),
        "svm_mnist": dict(optimizer="SMO and primal Adam", kernel_gamma=1.0, linear_lr=0.01),
        "boosting_mnist": dict(
            optimizer="CART residual fitting", inputs="49 top-half pixel averages"
        ),
        "gmm_mnist": dict(optimizer="EM", covariance_floor=1e-5),
        "dimension_mnist": dict(optimizer="none: nearest-neighbour data analysis"),
        "tsne": dict(
            optimizer="momentum gradient descent",
            lr=50.0,
            momentum=0.7,
            fit_split="official test split used as the embedding dataset",
        ),
    }.get(name, {})
    settings = dict(
        version=VERSION,
        **{"seed": SEED, **optimization, **settings},
        source_sha256=realdata.source_hash(*source_files),
    )

    def arrays():
        from . import mnist

        values = {key: np.asarray(value) for key, value in build().items()}
        parameters = {
            k: v
            for k, v in values.items()
            if "weights" in k
            or k.startswith(
                (
                    "plain_",
                    "minibatch_",
                    "rbf",
                    "linear_",
                    "lifted_",
                    "beta4_",
                    "annealed_",
                    "a_",
                    "b_",
                )
            )
            or k in ("coefficients", "trees", "bag_trees", "fast_trees", "outer_alpha", "maps")
        }
        dataset = dict(
            dataset="MNIST",
            source_url=mnist.URL,
            raw_sha256=mnist.SHA256,
            selection="Source indices and derived training arrays are stored in this artifact.",
            partitions="Official train/test partitions; t-SNE fits its displayed embedding data.",
            preprocessing="PCA fitted on training only; pixels / 255 before mean pooling.",
        )
        evidence = realdata.save_evidence(
            "vision-" + name, parameters, values, dataset, settings, source_files=source_files
        )
        values["evidence_json"] = np.array(json.dumps(evidence, sort_keys=True))
        values["run_settings"] = np.array(json.dumps(settings, sort_keys=True))
        return values

    values = cached_npz("next_vision_" + name, settings, arrays)
    RUN_EVIDENCE[name] = json.loads(str(values["evidence_json"]))
    return values


@lru_cache(maxsize=1)
def mnist_data():
    from pathlib import Path

    from .mnist import SHA256, load

    values = load()
    if sha256_of(Path(DEFAULT_CACHE) / "mnist.npz") != SHA256:
        raise ValueError("MNIST archive does not match the pinned dataset checksum")
    return values


def parameter_arrays(parameters, prefix="weights_"):
    """Persist the actual fitted arrays alongside the measurements they produced."""
    return {prefix + key: np.asarray(value) for key, value in parameters.items()}


def mnist_projection(n_train=1200, n_test=300, digits=tuple(range(10)), dimensions=2):
    """A fixed training-only PCA map; every target point remains a real MNIST image."""
    x, y, xt, yt = mnist_data()
    train_ids = np.flatnonzero(np.isin(y, digits))[:n_train]
    test_ids = np.flatnonzero(np.isin(yt, digits))[:n_test]
    train_pixels, test_pixels = vm.downsample(x[train_ids]), vm.downsample(xt[test_ids])
    mean = train_pixels.mean(0)
    _, _, axes = np.linalg.svd(train_pixels - mean, full_matrices=False)
    components = axes[:dimensions]
    scale = max(float(np.std((train_pixels - mean) @ components[:2].T)), 1e-8)
    return dict(
        train=(train_pixels - mean) @ components.T / scale,
        test=(test_pixels - mean) @ components.T / scale,
        train_labels=y[train_ids],
        test_labels=yt[test_ids],
        train_indices=train_ids,
        test_indices=test_ids,
        pca_mean=mean,
        pca_components=components,
        pca_scale=np.array(scale),
        train_pixels=train_pixels,
        test_pixels=test_pixels,
    )


def projection_arrays(data):
    return {"data_" + key: value for key, value in data.items()}


def gmm_log_likelihood(x, model):
    logs = []
    for weight, mean, cov in zip(model["pi"], model["mu"], model["cov"], strict=True):
        delta = x - mean
        logs.append(
            np.log(weight)
            - 0.5
            * (
                np.einsum("ni,ij,nj->n", delta, np.linalg.inv(cov), delta)
                + np.linalg.slogdet(cov)[1]
                + x.shape[1] * np.log(2 * np.pi)
            )
        )
    return float(np.logaddexp.reduce(np.array(logs), axis=0).mean())


def normalize(x):
    return np.asarray(x) / max(float(np.max(np.abs(x))), 1e-12)


def receptive_field_sizes(kernels, strides, dilations):
    field, jump, fields = 1, 1, [1]
    for kernel, stride, dilation in zip(kernels, strides, dilations, strict=True):
        field += (kernel - 1) * dilation * jump
        jump *= stride
        fields.append(field)
    return np.array(fields)


def receptive_field():
    from . import mnist

    p, _ = mnist.cnn_train()
    _, _, x, y = mnist_data()
    f = mnist._cnn_forward(p, x[:1])
    conv = np.zeros((28, 28))
    conv[12:15, 12:15] = np.abs(p[0][0][:, :, 0])
    pooled = np.zeros((28, 28))
    q = int(f["arg"][0, 6, 6, 0])
    r, c = 12 + q // 2, 12 + q % 2
    pooled[r : r + 3, c : c + 3] = np.abs(p[0][0][:, :, 0])
    dg = np.eye(10)[[y[0]]]
    full = np.abs(vm.cnn_input_grad(p, x[:1], dg)[0].reshape(28, 28))
    return result(
        "receptive-field",
        "MNIST CNN: measured input gradients and exact support",
        [
            images(
                "One image, one convolution",
                "A 3x3 convolution reads nine input locations.",
                [x[0].reshape(28, 28), normalize(conv)],
                ["MNIST input", "conv support"],
            ),
            bars(
                "Depth expands the possible support",
                "Kernel and stride determine theoretical receptive fields.",
                np.r_[
                    receptive_field_sizes([3, 2, 3], [1, 2, 1], [1, 1, 1]),
                    receptive_field_sizes([3, 3], [1, 1], [2, 2])[-1],
                    28,
                ],
                ["input", "conv 3", "pool 2", "next conv 3", "two dilated", "dense"],
            ),
            images(
                "Pooling selects a local path",
                "The max-pool gradient follows the winning convolution cell.",
                [normalize(conv), normalize(pooled), normalize(full)],
                ["conv", "pool", "logit"],
            ),
            matrix(
                "Effective and theoretical fields",
                "Within the theoretical support, gradient magnitudes are uneven.",
                normalize(full),
                ("nonzero pixels: %d / 784" % np.count_nonzero(full),),
            ),
        ],
    )


def vit_attention():
    def build():
        x, y, xt, yt = mnist_data()
        rng = np.random.default_rng(SEED)
        p = vm.small_vit_init(rng, 2)
        opt, losses = Adam(p, 0.002), []
        for _ in range(700):
            ids = rng.choice(10000, 32)
            loss, g, _ = vm.small_vit_loss(p, x[ids], y[ids])
            opt.step(p, g)
            losses.append(loss)
        _, _, (logits, att) = vm.small_vit_loss(p, xt[:128], yt[:128], False)
        return dict(
            loss=losses,
            attention=att.mean(1),
            accuracy=np.mean(logits.argmax(1) == yt[:128]),
            example=xt[0],
            example_attention=att[:, 0],
            train_indices=np.arange(10000),
            test_indices=np.arange(128),
            **parameter_arrays(p),
        )

    d = cached("vit_two_layer", build, layers=2, width=16, heads=2, steps=700, subset=10000)
    from .vit import patches

    coords = np.array([(i // 4, i % 4) for i in range(16)])
    distance = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
    att = d["attention"]
    reach = (att[:, :, 1:, 1:] * distance).sum((1, 2, 3)) / att[:, :, 1:, 1:].sum((1, 2, 3))
    return result(
        "vit-attention",
        "A two-block vision transformer trained on real MNIST patches",
        [
            images(
                "Sixteen patches become tokens",
                "The model embeds 7x7 patches and adds a learned CLS token.",
                patches(d["example"][None])[0].reshape(16, 7, 7),
                [str(i) for i in range(16)],
            ),
            matrix(
                "First block: CLS reads the image",
                "Rows are two attention heads; columns are sixteen image patches.",
                d["example_attention"][0, :, 0, 1:],
            ),
            matrix(
                "Second block: refined attention",
                "Both blocks were trained together with the image classification objective.",
                d["example_attention"][1, :, 0, 1:],
            ),
            bars(
                "Spatial reach by depth",
                "Distance is measured in patch-grid units over 128 held-out images.",
                reach,
                ["block 1", "block 2"],
                ("held-out accuracy %.1f%%" % (100 * d["accuracy"]),),
            ),
        ],
        scope="Two-block width-16 ViT; 700 updates; measured accuracy, no convergence claim",
    )


def clip_contrastive():
    def build():
        x, y, xt, yt = mnist_data()
        z, zt = vm.downsample(x[:12000]), vm.downsample(xt[:1000])
        p, loss, snaps = vm.fit_clip(z, y[:12000])
        _, _, s = vm.info_nce(p, zt, yt[:1000])
        return dict(
            loss=loss,
            snaps=snaps,
            predictions=s.argmax(1),
            targets=yt[:1000],
            scores=s,
            train_indices=np.arange(12000),
            test_indices=np.arange(1000),
            **parameter_arrays(p),
        )

    d = cached("clip_symmetric", build, steps=900, subset=12000, resolution=7, width=16, seed=53)
    temp = np.array([0.03, 0.07, 0.15, 0.4, 1.0])
    score = d["scores"][0] * 0.15
    entropy = [
        -np.sum((pr := toy.softmax(score[None] / t))[0] * np.log(pr[0] + 1e-20)) for t in temp
    ]
    return result(
        "clip-contrastive",
        "7x7 MNIST images paired with synthetic digit-word tokens",
        [
            matrix(
                "Before contrastive training",
                "Rows are one image per digit; columns are digit-word prototypes.",
                d["snaps"][0],
            ),
            curve(
                "Shared space learned from pairs",
                "Image and word encoders train together with contrastive cross-entropy.",
                d["loss"],
                ["InfoNCE"],
            ),
            matrix(
                "After contrastive training",
                "A matching pair competes against nine digit-word alternatives.",
                d["snaps"][-1],
            ),
            bars(
                "Temperature sharpens the distribution",
                "Held-out images are compared with the same digit labels used during training.",
                entropy,
                [".03", ".07", ".15", ".4", "1.0"],
                ("test accuracy %.1f%%" % (100 * np.mean(d["predictions"] == d["targets"])),),
            ),
        ],
        scope="Linear encoders; fixed digit-name labels; 7x7 downsampled MNIST",
        seed=53,
    )


def adversarial_examples():
    from . import mnist

    def build():
        params, _ = mnist.train()
        xt, yt, x, y = mnist_data()
        x, y = x[:160], y[:160]
        eps = np.array([0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3])
        fgsm, pgd = [], []
        clean = mnist._forward(params, x)[2]
        acts, pre, pr = mnist._forward(params, x)
        grad = mnist._backward(params, acts, pre, pr, np.eye(10)[y])[2][0]
        examples = []
        for e in eps:
            a = np.clip(x + e * np.sign(grad), 0, 1)
            fgsm.append(np.mean(mnist._forward(params, a)[2].argmax(1) == y))
            z = x.copy()
            for _ in range(12):
                ac, pe, pp = mnist._forward(params, z)
                gg = mnist._backward(params, ac, pe, pp, np.eye(10)[y])[2][0]
                z = np.clip(z + e / 4 * np.sign(gg), x - e, x + e).clip(0, 1)
            pgd.append(np.mean(mnist._forward(params, z)[2].argmax(1) == y))
            examples.append(z[0])
        tuned = {f"{q}{i}": a.copy() for i, layer in enumerate(params) for q, a in zip("Wb", layer)}
        opt, rng = Adam(tuned, 0.0005), np.random.default_rng(SEED)
        for _ in range(220):
            ids = rng.choice(6000, 64)
            xx, yy = xt[ids], yt[ids]
            pp = mnist._as_layers(tuned, 3)
            ac, pe, pr = mnist._forward(pp, xx)
            gg = mnist._backward(pp, ac, pe, pr, np.eye(10)[yy])[2][0]
            adv = np.clip(xx + 0.1 * np.sign(gg), 0, 1)
            opt.step(
                tuned, mnist._mlp_grads(tuned, np.vstack((xx, adv)), np.eye(10)[np.tile(yy, 2)])
            )
        pp = mnist._as_layers(tuned, 3)
        ac, pe, pr = mnist._forward(pp, x)
        gg = mnist._backward(pp, ac, pe, pr, np.eye(10)[y])[2][0]
        defended = np.mean(
            mnist._forward(pp, np.clip(x + 0.1 * np.sign(gg), 0, 1))[2].argmax(1) == y
        )
        return dict(
            eps=eps,
            fgsm=fgsm,
            pgd=pgd,
            clean=x[0],
            examples=examples,
            p=clean[0],
            q=mnist._forward(params, np.array(examples)[3:4])[2][0],
            defense=[np.mean(pr.argmax(1) == y), defended],
            train_indices=np.arange(6000),
            test_indices=np.arange(160),
            **parameter_arrays(tuned),
            **parameter_arrays(
                {f"{q}{i}": a for i, layer in enumerate(params) for q, a in zip("Wb", layer)},
                "base_weights_",
            ),
        )

    d = cached("adversarial", build, test=160, pgd=12, fine_tuning_steps=220)
    return result(
        "adversarial-examples",
        "Held-out MNIST attacks on the existing trained MLP",
        [
            images(
                "Small perturbations, changed evidence",
                "PGD maximises cross-entropy inside an L-infinity budget.",
                [
                    d["clean"].reshape(28, 28),
                    d["examples"][3].reshape(28, 28),
                    np.abs(d["examples"][3] - d["clean"]).reshape(28, 28) * 20,
                ],
                ["clean", "PGD .10", "delta x20"],
            ),
            matrix(
                "The prediction distribution changes",
                "The original and attacked image use exactly the same classifier.",
                np.vstack((d["p"], d["q"])),
            ),
            curve(
                "Accuracy against attack budget",
                "Both attacks are evaluated on the same 160 held-out images.",
                [d["fgsm"], d["pgd"]],
                ["FGSM", "12-step PGD"],
                d["eps"],
            ),
            bars(
                "A small adversarial fine-tune",
                "A separate 220-step fine-tune mixes clean and FGSM examples.",
                [d["fgsm"][0], d["defense"][0], d["fgsm"][3], d["defense"][1]],
                ["base clean", "tuned clean", "base FGSM .1", "tuned FGSM .1"],
                ("same budget; attacks recomputed after tuning",),
            ),
        ],
    )


def iou(box, boxes):
    lo = np.maximum(box[:2], boxes[:, :2])
    hi = np.minimum(box[2:], boxes[:, 2:])
    inter = np.maximum(hi - lo, 0).prod(1)
    area = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(1)
    return inter / np.maximum(area + areas - inter, 1e-12)


def nms(boxes, scores, threshold=0.4):
    order = np.argsort(-scores)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        order = order[1:][iou(boxes[i], boxes[order[1:]]) <= threshold]
    return np.array(keep, dtype=int)


def paint_boxes(image, boxes):
    out = np.repeat(image[..., None], 3, axis=-1)
    colors = [[1.0, 0.3, 0.1], [0.1, 0.7, 1.0]]
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = np.clip(np.round(box), 0, len(image) - 1).astype(int)
        out[y1 : y2 + 1, x1] = out[y1 : y2 + 1, x2] = colors[i % 2]
        out[y1, x1 : x2 + 1] = out[y2, x1 : x2 + 1] = colors[i % 2]
    return out


def object_detection():
    def build():
        x, y, xt, yt = mnist_data()
        xx, _, target = composed_digits(x, y, 1200, SEED)
        p, loss = toy.fit(
            xx.reshape(1200, -1),
            target.reshape(1200, -1) / 16,
            widths=(64, 32),
            steps=1100,
            lr=0.004,
        )
        im, _, truth = composed_digits(xt, yt, 80, SEED + 1)
        prediction = toy.forward(p, im.reshape(80, -1))[0].reshape(80, 2, 4) * 16
        offsets = np.array([[0, 0, 0, 0], [-1, 0, -1, 0], [1, 1, 1, 1]])
        boxes = (prediction[0, :, None] + offsets).reshape(-1, 4)
        scores = np.tile([0.9, 0.7, 0.6], 2)
        keep = nms(boxes, scores)
        overlap = np.array(
            [iou(truth[i, j], prediction[i, j : j + 1])[0] for i in range(80) for j in range(2)]
        )
        canvas = im[0, ..., 0]
        return dict(
            scene=canvas,
            proposals=paint_boxes(canvas, boxes),
            selected=paint_boxes(canvas, boxes[keep]),
            loss=loss,
            overlaps=overlap,
            keep=keep,
            boxes=boxes,
            train_images=xx,
            train_boxes=target,
            test_images=im,
            test_boxes=truth,
            **parameter_arrays(p),
        )

    d = cached("composed_detector", build, examples=1200, steps=1100, canvas=16, digit=7)
    return result(
        "object-detection-basics",
        "Two-object box regression on composed MNIST scenes",
        [
            images(
                "Two real digits, random positions",
                "The model learns both boxes from synthetic scenes with known coordinates.",
                [d["scene"]],
                ["two-digit scene"],
            ),
            curve(
                "Learning box coordinates",
                "A small regressor predicts both boxes from the composed pixels.",
                d["loss"],
                ["box MSE"],
            ),
            images(
                "Neighbouring proposals overlap",
                "Anchor offsets around both predicted boxes create duplicate proposals.",
                [d["proposals"]],
                ["six proposals"],
            ),
            images(
                "Non-maximum suppression",
                "Higher-scoring proposals suppress alternatives above the IoU threshold.",
                [d["selected"]],
                ["selected boxes"],
                (
                    "mean held-out IoU %.3f" % d["overlaps"].mean(),
                    "kept %d / %d proposals" % (len(d["keep"]), len(d["boxes"])),
                ),
            ),
        ],
        scope="Two known object slots; explicit anchor perturbations and NMS; 16x16 MNIST scenes",
    )


def composed_digits(x, y, n, seed):
    """Two disjoint 7x7 digits in a 16x16 canvas with exact pixel labels."""
    rng = np.random.default_rng(seed)
    source = vm.downsample(x)
    ones, sevens = np.flatnonzero(y == 1), np.flatnonzero(y == 7)
    im, truth, boxes = [], [], []
    for _ in range(n):
        canvas, mask = np.zeros((16, 16)), np.zeros((16, 16), dtype=int)
        pair = []
        for k, pool in enumerate((ones, sevens)):
            digit = source[rng.choice(pool)].reshape(7, 7)
            r = int(rng.integers(0, 10))
            c = int(rng.integers(0, 2)) if k == 0 else int(rng.integers(8, 10))
            canvas[r : r + 7, c : c + 7] = digit
            mask[r : r + 7, c : c + 7] = np.where(digit > 0.14, k + 1, 0)
            pair.append([c, r, c + 7, r + 7])
        im.append(canvas)
        truth.append(mask)
        boxes.append(pair)
    return np.array(im)[..., None], np.array(truth), np.array(boxes)


def segmentation():
    def build():
        x, y, xt, yt = mnist_data()
        rng = np.random.default_rng(SEED)
        small, yy, _ = composed_digits(x, y, 800, SEED)
        p, losses = vm.unet_init(rng), []
        opt = Adam(p, 0.006)
        for _ in range(1000):
            idx = rng.choice(len(small), 16)
            loss, g, _ = vm.unet_loss(p, small[idx], yy[idx])
            opt.step(p, g)
            losses.append(loss)
        im, truth, _ = composed_digits(xt, yt, 80, SEED + 1)
        _, _, prob = vm.unet_loss(p, im, truth)
        pr = prob.argmax(-1)
        ios = np.array(
            [
                np.sum((pr == c) & (truth == c)) / max(np.sum((pr == c) | (truth == c)), 1)
                for c in range(3)
            ]
        )
        return dict(
            loss=losses,
            images=im[:3, ..., 0],
            truth=truth[:3],
            pred=pr[:3],
            iou=ios,
            train_images=small,
            train_masks=yy,
            test_images=im,
            test_masks=truth,
            **parameter_arrays(p),
        )

    d = cached(
        "unet_composed_segment",
        build,
        canvas=16,
        digit_resolution=7,
        examples=800,
        steps=1000,
        classes=[0, 1, 7],
    )
    return result(
        "semantic-segmentation",
        "A trainable U-Net on composed two-digit MNIST scenes",
        [
            images(
                "Every pixel receives a label",
                "Synthetic scenes contain two real MNIST digits at random positions.",
                d["images"],
                ["input 1", "input 2", "input 3"],
            ),
            curve(
                "The encoder and decoder learn jointly",
                "Downsampled features return through upsampling and a high-resolution skip.",
                d["loss"],
                ["pixel cross-entropy"],
            ),
            images(
                "Held-out dense predictions",
                "These masks use scenes made from unseen MNIST test images.",
                d["pred"] / 2,
                ["mask 1", "mask 2", "mask 3"],
            ),
            bars(
                "Intersection over union per class",
                "Intersection is divided by union over eighty held-out scenes.",
                d["iou"],
                ["background", "one", "seven"],
            ),
        ],
        scope="One-level convolutional U-Net on 16x16 two-digit scenes, three pixel classes",
    )


def batchnorm():
    from . import mnist

    p, _ = mnist.cnn_train()
    x, _, xt, _ = mnist_data()
    train = mnist._cnn_forward(p, x[:256])["z1"]
    test = mnist._cnn_forward(p, xt[:32])["z1"]
    mu, var = train.mean((0, 1, 2)), train.var((0, 1, 2))
    out = (test - mu) / np.sqrt(var + 1e-5)
    tiny_mu, tiny_var = test[:1].mean((0, 1, 2)), test[:1].var((0, 1, 2))
    tiny = (test - tiny_mu) / np.sqrt(tiny_var + 1e-5)
    return result(
        "batchnorm-in-cnns",
        "Post-hoc normalization of a fixed MNIST CNN; no BatchNorm training",
        [
            matrix(
                "Channels have different statistics",
                "Mean and variance are measured over batch and spatial axes.",
                np.vstack((mu, var)),
            ),
            matrix(
                "Training and evaluation statistics",
                "Evaluation reuses training-set statistics instead of the current batch.",
                np.vstack((mu, test.mean((0, 1, 2)), tiny_mu)),
            ),
            bars(
                "A one-image batch changes the answer",
                "One-image statistics differ from a fixed training-image reference.",
                np.mean(np.abs(tiny - out), axis=(0, 1, 2)),
                ["channel 1", "channel 2", "channel 3", "channel 4"],
            ),
            images(
                "Normalised feature maps",
                "Fixed versus one-image statistics; a formula demonstration.",
                [normalize(out[0, :, :, 0]), normalize(tiny[0, :, :, 0])],
                ["fixed", "tiny batch"],
            ),
        ],
        scope="BatchNorm applied to trained CNN features; original classifier has no BatchNorm",
    )


def style_transfer():
    def build():
        from . import mnist

        p, _ = mnist.cnn_train()
        _, _, x, _ = mnist_data()
        content, style = x[0:1], x[18:19]
        sf = mnist._cnn_forward(p, style)["a1"].reshape(-1, 4)
        cf = mnist._cnn_forward(p, content)["a1"].reshape(-1, 4)
        gram = sf.T @ sf / len(sf)
        z = content.copy()
        trace, snaps = [], []
        for step in range(301):
            l, g, _ = vm.cnn_feature_grad(p, z, 0, gram, cf, 0.08)
            if step % 100 == 0:
                snaps.append(z[0].reshape(28, 28).copy())
            trace.append(l)
            if step < 300:
                z = np.clip(z - 0.05 * g / max(np.linalg.norm(g), 1e-12), 0, 1)
        ff = mnist._cnn_forward(p, z)["a1"].reshape(-1, 4)
        return dict(
            content=content.reshape(28, 28),
            style=style.reshape(28, 28),
            snaps=snaps,
            loss=trace,
            gram=gram,
            final=ff.T @ ff / len(ff),
            optimized_pixels=z,
            **parameter_arrays(
                {f"{q}{i}": a for i, layer in enumerate(p) for q, a in zip("Wb", layer)}
            ),
        )

    d = cached("style", build, steps=300, content_weight=0.08)
    return result(
        "style-transfer",
        "Optimising pixels through the existing four-channel MNIST CNN",
        [
            images(
                "Content and style references",
                "Content uses feature activations; style uses channel correlations.",
                [d["content"], d["style"]],
                ["content", "style"],
            ),
            matrix(
                "The style Gram matrix",
                "Each entry measures the mean product of two feature channels.",
                d["gram"],
            ),
            images(
                "Pixels move under gradient descent",
                "A content term keeps structure while correlations approach the style target.",
                d["snaps"],
                ["start", "100", "200", "300"],
            ),
            curve(
                "Measured style-and-content objective",
                "This small digit CNN produces modest changes, not natural-image textures.",
                d["loss"],
                ["objective"],
            ),
        ],
    )


def gan_training():
    def build():
        data = mnist_projection()
        real = data["train"]
        plain, ploss, surface, pp = vm.gan_run(
            real, minibatch=False, return_discriminator=True, return_parameters=True
        )
        mb, mloss, msurface, mp = vm.gan_run(
            real, minibatch=True, return_discriminator=True, return_parameters=True
        )
        centres = np.array([real[data["train_labels"] == k].mean(0) for k in range(10)])
        counts = [
            np.bincount(((z[:, None] - centres[None]) ** 2).sum(-1).argmin(1), minlength=10)
            for z in (plain[-1], mb[-1])
        ]
        return dict(
            real=real[:240],
            labels=data["train_labels"][:240],
            plain=plain,
            minibatch=mb,
            counts=counts,
            loss=np.vstack((ploss, mloss)),
            surface=surface,
            minibatch_surface=msurface,
            centres=centres,
            **projection_arrays(data),
            **parameter_arrays(pp, "plain_"),
            **parameter_arrays(mp, "minibatch_"),
        )

    d = cached(
        "gan_mnist", build, steps=1800, width=32, target="MNIST training-only 2D PCA", model_seed=51
    )
    return result(
        "gan-training",
        "NumPy GANs trained on 1,200 real MNIST images in training-only PCA coordinates",
        [
            points(
                "Measured MNIST target distribution",
                "Each point is one digit image; colour is its real label, not a fitted mode.",
                d["real"],
                d["labels"],
            ),
            images(
                "A generator learns the projected distribution",
                "Fixed noise passed through the trained generator at recorded updates.",
                [vm.scientific_canvas(points=z) for z in d["plain"]],
                ["update 0", "600", "1200", "1800"],
            ),
            points(
                "A minibatch-distance discriminator",
                "A second genuine training run sees batch diversity; improvement is not assumed.",
                d["minibatch"][-1],
            ),
            matrix(
                "Generated samples near each class centroid",
                "Rows: plain and minibatch GAN. Columns: nearest training class centroid 0–9.",
                d["counts"],
                (
                    "Centroid bins do not classify generated images.",
                    "This models 2D projections, not full digit images.",
                ),
            ),
        ],
        seed=51,
        scope="MNIST projection density; 1800 updates per GAN; points are not digit images",
    )


def grid_lines():
    axis = np.linspace(-2.0, 2.0, 60)
    return [np.column_stack((axis, np.full_like(axis, k))) for k in np.linspace(-2, 2, 7)] + [
        np.column_stack((np.full_like(axis, k), axis)) for k in np.linspace(-2, 2, 7)
    ]


def normalizing_flows():
    def build():
        data = mnist_projection()
        x = data["train"]
        rng = np.random.default_rng(SEED)
        p = toy.flow_init(rng)
        opt, loss = Adam(p, 0.003), []
        z = rng.normal(size=(240, 2))
        start = toy.flow_inverse(p, z)
        for _ in range(1800):
            ids = rng.choice(len(x), 128)
            l, g = toy.flow_loss_grads(p, x[ids])
            opt.step(p, g)
            loss.append(l)
        encoded, ld, _ = toy.flow_forward(p, x[:240])
        return dict(
            x=x[:240],
            labels=data["train_labels"][:240],
            start=start,
            final=toy.flow_inverse(p, z),
            encoded=encoded,
            logdet=ld,
            loss=loss,
            error=np.max(np.abs(toy.flow_inverse(p, encoded) - x[:240])),
            nll=[toy.flow_loss_grads(p, x)[0], toy.flow_loss_grads(p, data["test"])[0]],
            grid=np.array([toy.flow_inverse(p, line) for line in grid_lines()]),
            **parameter_arrays(p),
            **projection_arrays(data),
        )

    d = cached(
        "flow_mnist", build, layers=6, hidden=24, steps=1800, target="MNIST training-only 2D PCA"
    )
    return result(
        "normalizing-flows",
        "Six affine coupling layers fitted to real MNIST training-only PCA coordinates",
        [
            points(
                "A density of measured digit projections",
                "PCA is fitted on training images only; digit labels are not used by the flow.",
                d["x"],
                d["labels"],
            ),
            images(
                "The inverse transforms a grid",
                "Grid lines and Gaussian samples use the same fitted algebraic inverse.",
                [
                    vm.scientific_canvas(paths=grid_lines()),
                    vm.scientific_canvas(points=d["final"], paths=d["grid"]),
                ],
                ["base grid", "learned inverse"],
            ),
            curve(
                "Actual likelihood optimization",
                "Every value is measured during 1,800 NumPy Adam updates.",
                d["loss"],
                ["training NLL"],
            ),
            bars(
                "Evaluate the same fitted density",
                "Mean negative log density in fixed PCA coordinates; lower is better.",
                d["nll"],
                ["training images", "held-out images"],
                ("max round-trip error %.3g" % d["error"], "Densities refer to 2D coordinates."),
            ),
        ],
        seed=SEED,
    )


def few_step_errors(original, rectified, start, end, rect_end):
    out = []
    for model, reference in ((original, end), (rectified, rect_end)):
        errors = []
        for steps in (1, 4, 10):
            z = start.copy()
            for k in range(steps):
                z += toy.forward(model, np.column_stack((z, np.full(len(z), k / steps))))[0] / steps
            errors.append(np.mean((z - reference) ** 2))
        out.append(errors)
    return np.array(out)


def flow_matching():
    def build():
        data = mnist_projection(1800, 300)
        x, y = data["train"], data["train_labels"]
        rng = np.random.default_rng(SEED)
        p = toy.init((3, 48, 48, 2), rng)
        opt, loss = Adam(p, 0.003), []
        for _ in range(2200):
            batch = x[rng.choice(len(x), 96)]
            noise = rng.normal(size=batch.shape)
            t = rng.random((len(batch), 1))
            current = (1 - t) * noise + t * batch
            l, g = toy.loss_grads(p, np.column_stack((current, t)), batch - noise)
            opt.step(p, g)
            loss.append(l)
        start = rng.normal(size=(180, 2))
        trails = [start.copy()]
        z = start.copy()
        for k in range(40):
            z += toy.forward(p, np.column_stack((z, np.full(len(z), k / 40))))[0] / 40
            trails.append(z.copy())
        # Rectification reuses endpoint couplings; one genuine fine-tune.
        rect = {k: v.copy() for k, v in p.items()}
        ropt = Adam(rect, 0.001)
        for _ in range(400):
            ids = rng.choice(len(z), 96)
            t = rng.random((96, 1))
            inp = (1 - t) * start[ids] + t * z[ids]
            _, g = toy.loss_grads(rect, np.column_stack((inp, t)), z[ids] - start[ids])
            ropt.step(rect, g)
        rz, rtr = start.copy(), [start.copy()]
        for k in range(40):
            rz += toy.forward(rect, np.column_stack((rz, np.full(len(rz), k / 40))))[0] / 40
            rtr.append(rz.copy())

        def excess(tr):
            return np.mean(
                np.linalg.norm(np.diff(tr, axis=0), axis=-1).sum(0)
                - np.linalg.norm(tr[-1] - tr[0], axis=-1)
            )

        return dict(
            target=x[:180],
            labels=y[:180],
            start=start,
            final=z,
            rect=rz,
            loss=loss,
            excess=[excess(np.array(trails)), excess(np.array(rtr))],
            paths=np.array(trails),
            rect_paths=np.array(rtr),
            few_error=few_step_errors(p, rect, start, z, rz),
            **parameter_arrays(p),
            **parameter_arrays(rect, "rectified_weights_"),
            **projection_arrays(data),
        )

    d = cached(
        "flow_matching_mnist",
        build,
        steps=2200,
        rectification=400,
        ode_steps=40,
        target="MNIST training-only 2D PCA",
    )
    return result(
        "flow-matching",
        "Learned velocity from Gaussian noise to real MNIST training-only PCA coordinates",
        [
            points(
                "Noise is the starting distribution",
                "A time-conditioned MLP learns data minus noise along sampled bridges.",
                d["start"],
            ),
            curve(
                "Velocity regression learns a flow",
                "The regression target is the exact velocity of each straight training path.",
                d["loss"],
                ["velocity MSE"],
            ),
            images(
                "Learned paths and rectified paths",
                "Both models integrate a learned velocity field from the same initial noise.",
                [
                    vm.scientific_canvas(
                        points=d["final"], paths=d["paths"][:, :12].transpose(1, 0, 2)
                    ),
                    vm.scientific_canvas(
                        points=d["rect"], paths=d["rect_paths"][:, :12].transpose(1, 0, 2)
                    ),
                ],
                ["original", "rectified"],
                ("four-step endpoint MSE %.3g / %.3g" % tuple(d["few_error"][:, 1]),),
            ),
            bars(
                "Rectification changes path curvature",
                "A second fit learns straight couplings between generated endpoints.",
                d["excess"],
                ["original", "rectified"],
                ("mean path length minus endpoint distance",),
            ),
        ],
    )


def classifier_free_guidance():
    def build():
        x, y, xt, yt = mnist_data()
        small = vm.downsample(x[:16000])
        p, loss, abar = vm.conditional_diffusion(small, y[:16000])
        samples = vm.conditional_sample(p, abar)
        classifier, _ = toy.fit(
            small, y[:16000], widths=(64,), steps=900, lr=0.005, classification=True
        )
        test = toy.forward(classifier, vm.downsample(xt[:1000]))[0].argmax(1)
        labels = np.repeat([0, 3, 5, 7], 8)
        accuracy = [
            np.mean(toy.forward(classifier, batch)[0].argmax(1) == labels) for batch in samples
        ]
        diversity = samples.reshape(4, 4, 8, 49).var(axis=2).mean((1, 2))
        return dict(
            loss=loss,
            samples=samples[:, [0, 8, 16, 24]],
            diversity=diversity,
            accuracy=accuracy,
            classifier_accuracy=np.mean(test == yt[:1000]),
            schedule_abar=abar,
            train_indices=np.arange(16000),
            test_indices=np.arange(1000),
            **parameter_arrays(p),
            **parameter_arrays(classifier, "classifier_weights_"),
        )

    d = cached(
        "cfg_diversity",
        build,
        resolution=7,
        steps=2200,
        widths=[96, 96],
        T=80,
        dropout=0.15,
        samples_per_class=8,
        classifier_steps=900,
        seed=62,
        sampling_seed=63,
        classifier_seed=51,
    )
    return result(
        "classifier-free-guidance",
        "Conditional diffusion on 7x7 MNIST; measured guidance sweep",
        [
            curve(
                "Learning with missing conditions",
                "Fifteen percent of minibatch captions are dropped during noise prediction.",
                d["loss"],
                ["noise MSE"],
            ),
            images(
                "Unconditional and ordinary guidance",
                "Each row begins with the same noise and asks for digits 0, 3, 5 and 7.",
                d["samples"][:2].reshape(-1, 7, 7),
                ["w0"] * 4 + ["w1"] * 4,
            ),
            images(
                "Stronger guidance",
                "Guidance combines unconditional and conditional noise estimates.",
                d["samples"][2:].reshape(-1, 7, 7),
                ["w3"] * 4 + ["w8"] * 4,
            ),
            curve(
                "Fidelity proxy and within-class diversity",
                "Accuracy uses a tested classifier; diversity is within-class pixel variance.",
                [d["accuracy"], d["diversity"]],
                ["class match", "pixel variance"],
                np.array([0, 1, 3, 8]),
                (
                    "32 generated samples per scale",
                    "classifier test accuracy %.1f%%" % (100 * d["classifier_accuracy"]),
                ),
            ),
        ],
        scope="7x7 MNIST, eighty-step deterministic sampler, 8 samples per requested class",
        seed=62,
        sampling_seed=63,
        classifier_seed=51,
    )


def ddpm_ddim():
    from . import diffusion as df

    def build():
        p, _ = df.train()
        sched = df.schedule()
        _, _, abar = sched
        rng = np.random.default_rng(SEED)
        initial = rng.normal(size=(4, 784))
        results = []
        for steps in (10, 25, 50, df.T_STEPS):
            x = initial.copy()
            times = np.linspace(df.T_STEPS, 0, steps + 1).astype(int)
            for t, prev in itertools.pairwise(times):
                eps = df.forward(p, x, np.full(len(x), t))["eps"]
                x0 = df.predict_x0(x, eps, t, abar)
                x = np.sqrt(abar[prev]) * x0 + np.sqrt(1 - abar[prev]) * eps
            results.append(df.to_image(x))
        x = initial.copy()
        for t in range(df.T_STEPS, 0, -1):
            x, _, _ = df.reverse_step(p, x, t, rng.normal(size=x.shape), sched)
        return dict(ddim=results, ddpm=df.to_image(x), initial_noise=initial, **parameter_arrays(p))

    d = cached("ddim", build, schedule="existing 200-step cosine", steps=[10, 25, 50, 200])
    from . import mnist

    classifier, _ = mnist.train()
    scores = [mnist._forward(classifier, a)[2].max(1).mean() for a in d["ddim"]]
    return result(
        "ddpm-vs-ddim",
        "The same trained MNIST denoiser, stochastic or deterministic sampling",
        [
            images(
                "Stochastic reverse diffusion",
                "The repository denoiser was trained for 200 diffusion steps.",
                d["ddpm"].reshape(-1, 28, 28),
                ["DDPM"] * 4,
            ),
            images(
                "Ten deterministic steps",
                "DDIM jumps directly between selected noise levels using the same weights.",
                d["ddim"][0].reshape(-1, 28, 28),
                ["10 steps"] * 4,
            ),
            images(
                "Fifty deterministic steps",
                "The same starting noise makes changes in the sampler visible.",
                d["ddim"][2].reshape(-1, 28, 28),
                ["50 steps"] * 4,
            ),
            bars(
                "Classifier confidence is only a proxy",
                "Four generated samples per setting cannot establish distribution quality.",
                scores,
                ["10", "25", "50", "200"],
            ),
        ],
        scope="Existing 200-step model, not a retrained 1000-step denoiser",
    )


def vae_collapse():
    def build():
        x, _, xt, _ = mnist_data()
        x, xt = vm.downsample(x[:12000]), vm.downsample(xt[:256])
        all_kl, all_recon, saved = [], [], {}
        for anneal in (False, True):
            rng = np.random.default_rng(SEED)
            p = dict(
                E=rng.normal(0, 0.06, (49, 4)),
                eb=np.zeros(4),
                L=rng.normal(0, 0.01, (49, 4)),
                lb=np.zeros(4),
                D=rng.normal(0, 0.06, (4, 49)),
                db=np.zeros(49),
                AR=rng.normal(0, 0.02, (49, 49)),
            )
            opt, trace = Adam(p, 0.004), []
            for step in range(1800):
                xx = x[rng.choice(len(x), 96)]
                beta = min(1, step / 900) if anneal else 4.0
                _, g, (kl, _) = vm.autoregressive_vae_loss(
                    p, xx, rng.normal(size=(len(xx), 4)), beta
                )
                opt.step(p, g)
                trace.append(kl)
            _, _, (_, r) = vm.autoregressive_vae_loss(p, xt, np.zeros((len(xt), 4)))
            all_kl.append(trace)
            all_recon.append(r[:4])
            saved.update(parameter_arrays(p, "annealed_" if anneal else "beta4_"))
        return dict(
            kl=np.array(all_kl),
            recon=np.array(all_recon),
            train_indices=np.arange(12000),
            test_indices=np.arange(256),
            **saved,
        )

    d = cached(
        "vae_strong_decoder",
        build,
        steps=1800,
        resolution=7,
        beta=[4, "anneal to 1"],
        decoder="strictly causal 49x49 pixel matrix",
    )
    return result(
        "vae-posterior-collapse",
        "A VAE with a strong autoregressive pixel decoder",
        [
            curve(
                "High KL pressure",
                "Beta four penalises latent information; previous pixels can explain the image.",
                d["kl"][0].T,
                ["z1", "z2", "z3", "z4"],
            ),
            curve(
                "KL annealed during training",
                "The second run increases beta from zero to one over nine hundred updates.",
                d["kl"][1].T,
                ["z1", "z2", "z3", "z4"],
            ),
            images(
                "Teacher-forced decoder reconstructions",
                "The decoder sees earlier pixels plus the latent mean, never future pixels.",
                d["recon"].reshape(-1, 7, 7),
                ["beta4"] * 4 + ["annealed"] * 4,
            ),
            bars(
                "How much information survives?",
                "Final KL is measured; annealing is not guaranteed to prevent collapse.",
                d["kl"][:, -50:].mean((1, 2)),
                ["beta4", "annealed"],
            ),
        ],
        scope="Four latent dimensions, causal linear autoregressive decoder, 7x7 MNIST",
    )


def vq_vae():
    def build():
        x, _, xt, _ = mnist_data()
        x, xt = vm.downsample(x[:12000]), vm.downsample(xt[:256])
        rng = np.random.default_rng(SEED)
        p = dict(
            E=rng.normal(0, 0.07, (49, 2)),
            eb=np.zeros(2),
            C=rng.normal(0, 0.2, (24, 2)),
            D=rng.normal(0, 0.07, (2, 49)),
            db=np.zeros(49),
        )
        opt, losses = Adam(p, 0.004), []
        for _ in range(1800):
            xx = x[rng.choice(len(x), 96)]
            l, g, _ = vm.vq_loss_grads(p, xx)
            opt.step(p, g)
            losses.append(l)
            if len(losses) % 100 == 0 and len(losses) < 900:
                enc = np.tanh(x[:1000] @ p["E"] + p["eb"])
                used = ((enc[:, None] - p["C"][None]) ** 2).sum(-1).argmin(1)
                counts = np.bincount(used, minlength=len(p["C"]))
                dead = np.flatnonzero(counts == 0)
                p["C"][dead] = enc[rng.choice(len(enc), len(dead))]
        _, _, (h, z, ids, r) = vm.vq_loss_grads(p, xt)
        return dict(
            loss=losses,
            h=h,
            z=z,
            code=p["C"],
            counts=np.bincount(ids, minlength=24),
            recon=r[:4],
            original=xt[:4],
            train_indices=np.arange(12000),
            test_indices=np.arange(256),
            **parameter_arrays(p),
        )

    d = cached("vq_restart", build, steps=1800, codes=24, latent=2, resolution=7)
    return result(
        "vq-vae-codebook",
        "A genuinely quantised autoencoder trained on 7x7 MNIST",
        [
            points(
                "Encoder outputs approach code vectors",
                "The decoder reads the nearest code. A straight-through gradient trains encoding.",
                d["h"],
            ),
            points(
                "Twenty-four learned code vectors",
                "Each encoder output snaps to its Euclidean nearest neighbour.",
                d["code"],
            ),
            images(
                "Quantised reconstruction",
                "Only the discrete code reaches the decoder.",
                np.vstack((d["original"], d["recon"])).reshape(-1, 7, 7),
                ["input"] * 4 + ["output"] * 4,
            ),
            bars(
                "Code usage and dead entries",
                "Dead codes are restarted early; usage is measured on 256 held-out images.",
                d["counts"],
                [str(i) for i in range(24)],
                ("unused codes: %d" % np.sum(d["counts"] == 0),),
            ),
        ],
    )


def autoregressive_pixels():
    def build():
        x, _, _, _ = mnist_data()
        x = vm.downsample(x[:12000]).reshape(-1, 7, 7, 1)
        rng = np.random.default_rng(SEED)
        p = dict(
            W=rng.normal(0, 0.1, (3, 3, 1, 16)),
            b=np.zeros(16),
            O=rng.normal(0, 0.1, (16, 1)),
            c=np.zeros(1),
        )
        opt, loss = Adam(p, 0.005), []
        for _ in range(1300):
            xx = x[rng.choice(len(x), 64)]
            l, g, _ = vm.masked_conv_loss(p, xx)
            opt.step(p, g)
            loss.append(l)
        generated, snaps = np.zeros((4, 7, 7, 1)), []
        for i in range(7):
            for j in range(7):
                _, _, r = vm.masked_conv_loss(p, generated)
                generated[:, i, j, 0] = rng.random(4) < r[:, i, j, 0]
            if i in (0, 2, 4, 6):
                snaps.append(generated[0, :, :, 0].copy())
        return dict(
            loss=loss,
            samples=generated[..., 0],
            snaps=snaps,
            train_indices=np.arange(12000),
            **parameter_arrays(p),
        )

    d = cached("pixelcnn", build, steps=1300, resolution=7, hidden=16, mask="type A 3x3")
    return result(
        "autoregressive-pixels",
        "A masked convolutional Bernoulli model of 7x7 MNIST",
        [
            matrix(
                "A mask prevents looking ahead",
                "The centre and all future pixels have zero convolution weight.",
                [[1, 1, 1], [1, 0, 0], [0, 0, 0]],
            ),
            curve(
                "Conditional pixel likelihood",
                "The model trains every pixel in parallel, with the causal mask fixed.",
                d["loss"],
                ["mean BCE"],
            ),
            images(
                "Generation must proceed in order",
                "Rows are sampled from top to bottom and left to right.",
                d["snaps"],
                ["row 1", "row 3", "row 5", "row 7"],
            ),
            images(
                "Four complete independent samples",
                "This local model sees a limited past and may miss global digit structure.",
                d["samples"],
                ["sample 1", "sample 2", "sample 3", "sample 4"],
            ),
        ],
    )


def bias_variance():
    def build():
        x, _, xt, _ = mnist_data()
        train = x[:800].reshape(-1, 28, 28)
        test = xt[:160].reshape(-1, 28, 28)
        # A measured regression task: predict right-half ink from left-half ink.
        a, b = train[:, :, :14].mean((1, 2)), train[:, :, 14:].mean((1, 2))
        at, bt = test[:, :, :14].mean((1, 2)), test[:, :, 14:].mean((1, 2))
        order = np.argsort(at)
        at, bt = at[order], bt[order]
        rng = np.random.default_rng(SEED)
        predictions, coefficients, sampled = [], [], []
        for degree in range(1, 7):
            pred, coef, ids_all = [], [], []
            for _ in range(120):
                ids = rng.integers(0, len(a), 80)
                fit = np.polynomial.Polynomial.fit(a[ids], b[ids], degree, domain=[0, 1]).convert()
                pred.append(fit(at))
                coef.append(np.pad(fit.coef, (0, 7 - len(fit.coef))))
                ids_all.append(ids)
            predictions.append(pred)
            coefficients.append(coef)
            sampled.append(ids_all)
        pred = np.array(predictions)
        mean_error = ((pred.mean(1) - bt) ** 2).mean(1)
        variance = pred.var(1).mean(1)
        mse = ((pred - bt) ** 2).mean((1, 2))
        return dict(
            x=at,
            targets=bt,
            predictions=pred,
            coefficients=coefficients,
            bootstrap_indices=sampled,
            train_x=a,
            train_targets=b,
            train_indices=np.arange(800),
            test_indices=order,
            mean_error=mean_error,
            variance=variance,
            mse=mse,
            closure=np.abs(mse - mean_error - variance),
        )

    d = cached(
        "bias_variance_mnist",
        build,
        train=800,
        test=160,
        bootstrap=120,
        sample_size=80,
        degrees=list(range(1, 7)),
    )
    return result(
        "bias-variance",
        "Actual bootstrap polynomial fits: predict right-half ink from left-half ink in MNIST",
        [
            curve(
                "Different training samples, different predictions",
                "Eight degree-three fits from resampled images; x is left-half ink.",
                d["predictions"][2, :8],
                ["fit %d" % i for i in range(8)],
                d["x"],
            ),
            curve(
                "Compare the mean prediction with measurements",
                "Targets are right-half ink of unseen MNIST images, not a known noiseless law.",
                [d["predictions"][i].mean(0) for i in (0, 2, 5)] + [d["targets"]],
                ["degree 1", "degree 3", "degree 6", "measured target"],
                d["x"],
            ),
            curve(
                "Decompose finite-test squared error",
                "Average squared error equals squared error of the mean plus prediction variance.",
                [d["mean_error"], d["variance"], d["mse"]],
                ["mean prediction error", "prediction variance", "average MSE"],
                np.arange(1, 7),
            ),
            bars(
                "Check the empirical identity",
                "This identity does not identify population bias or irreducible noise.",
                d["closure"],
                [str(i) for i in range(1, 7)],
                ("True conditional mean is unknown.",),
            ),
        ],
        seed=SEED,
        scope="Measured MNIST regression; squared mean error is not claimed to be population bias",
    )


def cross_validation():
    def build():
        raw, labels, raw_test, test_labels = mnist_data()
        ids = np.flatnonzero(np.isin(labels, [3, 8]))[:200]
        test_ids = np.flatnonzero(np.isin(test_labels, [3, 8]))[:200]
        rng = np.random.default_rng(SEED)
        ids = ids[rng.permutation(len(ids))]
        x, y = vm.downsample(raw[ids]), (labels[ids] == 8).astype(int)
        xt, yt = vm.downsample(raw_test[test_ids]), (test_labels[test_ids] == 8).astype(int)
        gammas = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0])
        scores = np.zeros((6, 5))
        outer = []
        selected = []
        outer_alpha = []

        def fit(a, target, gamma):
            return np.linalg.solve(toy.kernel(a, a, gamma) + 0.03 * np.eye(len(a)), 2 * target - 1)

        for g, gamma in enumerate(gammas):
            for fold in range(5):
                valid = np.arange(len(x)) % 5 == fold
                alpha = fit(x[~valid], y[~valid], gamma)
                scores[g, fold] = np.mean(
                    (toy.kernel(x[valid], x[~valid], gamma) @ alpha > 0) == y[valid]
                )
        for fold in range(5):
            valid = np.arange(len(x)) % 5 == fold
            idx = np.flatnonzero(~valid)
            acc = []
            for gamma in gammas:
                inner_scores = []
                for inner_fold in range(4):
                    iv = idx[np.arange(len(idx)) % 4 == inner_fold]
                    it = idx[np.arange(len(idx)) % 4 != inner_fold]
                    alpha = fit(x[it], y[it], gamma)
                    inner_scores.append(
                        np.mean((toy.kernel(x[iv], x[it], gamma) @ alpha > 0) == y[iv])
                    )
                acc.append(np.mean(inner_scores))
            gamma = gammas[np.argmax(acc)]
            selected.append(gamma)
            alpha = fit(x[~valid], y[~valid], gamma)
            outer_alpha.append(alpha)
            outer.append(np.mean((toy.kernel(x[valid], x[~valid], gamma) @ alpha > 0) == y[valid]))
        gamma = gammas[np.argmax(scores.mean(1))]
        alpha = fit(x, y, gamma)
        test_accuracy = np.mean((toy.kernel(xt, x, gamma) @ alpha > 0) == yt)
        # This PCA is display-only; classifiers use fixed pooled pixels in every fold.
        mean = x.mean(0)
        _, _, axes = np.linalg.svd(x - mean, full_matrices=False)
        return dict(
            x=x,
            y=y,
            xy=(x - mean) @ axes[:2].T,
            scores=scores,
            outer=outer,
            selected=selected,
            outer_alpha=outer_alpha,
            weights_alpha=alpha,
            selected_gamma=gamma,
            gammas=gammas,
            test_accuracy=test_accuracy,
            train_indices=ids,
            test_indices=test_ids,
            test_x=xt,
            test_y=yt,
            display_mean=mean,
            display_components=axes[:2],
        )

    d = cached(
        "cross_validation_mnist",
        build,
        train=200,
        test=200,
        digits=[3, 8],
        outer_folds=5,
        inner_folds=4,
        ridge=0.03,
    )
    nested, optimistic = float(d["outer"].mean()), float(d["scores"].max(0).mean())
    return result(
        "cross-validation",
        "Nested kernel-ridge selection on real MNIST threes and eights",
        [
            points(
                "Two measured digit classes",
                "Display-only PCA; every model fits the original 49 pooled pixel features.",
                d["xy"],
                d["y"],
            ),
            matrix(
                "Five held-out roles within the training set",
                "Each row marks its validation examples; original MNIST test images stay separate.",
                (np.arange(200)[None, :] % 5 == np.arange(5)[:, None]).astype(float),
            ),
            matrix(
                "Measured validation accuracy",
                "Rows sweep inverse kernel length scale; columns are held-out folds.",
                d["scores"],
            ),
            bars(
                "Nested outer-fold scores",
                "Inner folds choose gamma; the final refit is evaluated on untouched test images.",
                d["outer"],
                ["fold %d" % (i + 1) for i in range(5)],
                (
                    "optimism gap %+.1f pp" % (100 * (optimistic - nested)),
                    "external test accuracy %.1f%%" % (100 * d["test_accuracy"]),
                ),
            ),
        ],
        seed=SEED,
    )


def svm_kernels():
    def build():
        data = mnist_projection(200, 200, (3, 8))
        x = data["train"]
        xt = data["test"]
        y = (data["train_labels"] == 8).astype(int)
        yt = (data["test_labels"] == 8).astype(int)
        settings = [0.15, 1.0, 10.0, 100.0]
        models = [toy.svm(x, y, c=c, passes=8) for c in settings]
        linear = toy.linear_svm(x, y, c=10.0)
        lift = np.column_stack((x, np.sum(x * x, axis=1)))
        lift_test = np.column_stack((xt, np.sum(xt * xt, axis=1)))
        lifted = toy.linear_svm(lift, y, c=10.0)
        params = {}
        for i, m in enumerate(models):
            params.update(parameter_arrays(m, "rbf%d_" % i))
        return dict(
            x=x,
            y=y,
            lift=lift,
            support=[np.sum(m["a"] > 1e-6) for m in models],
            support_mask=models[2]["a"] > 1e-6,
            acc=[np.mean((toy.svm_predict(m, xt) > 0) == yt) for m in models],
            linear_accuracy=np.mean((xt @ linear["w"] + linear["b"] > 0) == yt),
            lifted_accuracy=np.mean((lift_test @ lifted["w"] + lifted["b"] > 0) == yt),
            **parameter_arrays(linear, "linear_"),
            **parameter_arrays(lifted, "lifted_"),
            **params,
            **projection_arrays(data),
        )

    d = cached(
        "svm_mnist",
        build,
        train=200,
        test=200,
        digits=[3, 8],
        cs=[0.15, 1.0, 10.0, 100.0],
        smo_passes=8,
        linear_updates=600,
    )
    return result(
        "svm-and-kernels",
        "Actual NumPy SVM fits on MNIST in a training-only two-coordinate projection",
        [
            points(
                "The same measured threes and eights",
                "Each point is a real training image; no synthetic rings or imposed separation.",
                d["x"],
                d["y"],
            ),
            images(
                "Add radius squared as a feature",
                "An oblique view of the explicit lift; held-out accuracy need not improve.",
                [
                    vm.scientific_canvas(
                        points=d["lift"][:, :2] + d["lift"][:, 2:3] * [0.25, 0.45],
                        classes=d["y"],
                        limits=(-3, 4, -3, 5),
                    )
                ],
                ["feature lift"],
                (
                    "linear test %.1f%%" % (100 * d["linear_accuracy"]),
                    "lifted test %.1f%%" % (100 * d["lifted_accuracy"]),
                ),
            ),
            points(
                "Measured support vectors",
                "These training points have nonzero dual coefficients in the fitted RBF model.",
                d["x"][d["support_mask"]],
                d["y"][d["support_mask"]],
            ),
            bars(
                "Support counts as C changes",
                "Counts describe the fitted models; accuracy below uses independent test digits.",
                d["support"],
                [".15", "1", "10", "100"],
                ("test accuracies: " + ", ".join("%.0f%%" % (100 * a) for a in d["acc"]),),
            ),
        ],
        seed=SEED,
        linear_accuracy=float(d["linear_accuracy"]),
        lifted_accuracy=float(d["lifted_accuracy"]),
    )


def gradient_boosting():
    def build():
        raw, _, rt, _ = mnist_data()
        images = raw[:500].reshape(-1, 28, 28)
        test = rt[:300].reshape(-1, 28, 28)
        # Predict missing bottom-half ink using only observed top-half pixels.
        x = images[:, :14].reshape(500, 7, 2, 7, 4).mean((2, 4)).reshape(500, 49)
        xt = test[:, :14].reshape(300, 7, 2, 7, 4).mean((2, 4)).reshape(300, 49)
        y = images[:, 14:].mean((1, 2))
        truth = test[:, 14:].mean((1, 2))
        rng = np.random.default_rng(SEED)
        pred = np.full(len(y), y.mean())
        held = np.full(len(xt), y.mean())
        bag = np.zeros(len(xt))
        train_loss = []
        boosted = []
        bagged = []
        fast = []
        trees = []
        bag_trees = []
        fast_trees = []
        fp = pred.copy()
        ft = held.copy()
        for _ in range(60):
            tree = toy.tree_fit(x, y - pred, depth=3)
            trees.append(tree)
            pred += 0.12 * toy.tree_predict(tree, x)
            held += 0.12 * toy.tree_predict(tree, xt)
            ids = rng.integers(0, len(x), len(x))
            bt = toy.tree_fit(x[ids], y[ids], depth=3)
            bag_trees.append(bt)
            bag += toy.tree_predict(bt, xt)
            tr = toy.tree_fit(x, y - fp, depth=3)
            fast_trees.append(tr)
            fp += toy.tree_predict(tr, x)
            ft += toy.tree_predict(tr, xt)
            train_loss.append(np.mean((pred - y) ** 2))
            boosted.append(np.mean((held - truth) ** 2))
            bagged.append(np.mean((bag / len(trees) - truth) ** 2))
            fast.append(np.mean((ft - truth) ** 2))
        return dict(
            train_loss=train_loss,
            boosted_loss=boosted,
            bag_loss=bagged,
            fast_loss=fast,
            train_x=x,
            train_y=y,
            test_x=xt,
            test_y=truth,
            train_indices=np.arange(500),
            test_indices=np.arange(300),
            initial_prediction=y.mean(),
            trees=np.array(json.dumps(trees)),
            bag_trees=np.array(json.dumps(bag_trees)),
            fast_trees=np.array(json.dumps(fast_trees)),
        )

    d = cached(
        "boosting_mnist",
        build,
        train=500,
        test=300,
        trees=60,
        depth=3,
        rates=[0.12, 1.0],
        target="bottom-half mean ink from top-half pixels",
    )
    return result(
        "gradient-boosting",
        "Fit actual MNIST bottom-half ink using only top-half image measurements",
        [
            points(
                "A measured image-completion regression",
                "x: top-half ink; y: bottom-half ink. The model uses 49 top-half features.",
                np.column_stack((d["train_x"].mean(1), d["train_y"])),
            ),
            curve(
                "Trees correct measured residuals",
                "Each depth-three tree contributes twelve percent of its fitted correction.",
                [d["train_loss"], d["boosted_loss"]],
                ["training MSE", "held-out MSE"],
            ),
            curve(
                "Boosting and independent bootstrap trees",
                "Both methods use depth-three trees and the same independent test images.",
                [d["boosted_loss"], d["bag_loss"]],
                ["boosting", "bagging"],
            ),
            bars(
                "Measured error after sixty trees",
                "All targets are real image measurements; no expected winner is imposed.",
                [d["boosted_loss"][-1], d["fast_loss"][-1], d["bag_loss"][-1]],
                ["boost rate .12", "boost rate 1", "bagging"],
            ),
        ],
        seed=SEED,
    )


def mixture_picture(x, run):
    colors = np.array(
        [
            [0.15, 0.6, 0.55],
            [0.85, 0.3, 0.2],
            [0.4, 0.3, 0.8],
            [0.7, 0.6, 0.15],
            [0.15, 0.4, 0.7],
            [0.7, 0.25, 0.5],
            [0.3, 0.6, 0.2],
            [0.5, 0.5, 0.5],
        ]
    )
    ellipses = []
    theta = np.linspace(0, 2 * np.pi, 65)
    unit = np.column_stack((np.cos(theta), np.sin(theta)))
    for mu, cov in zip(run["mu"], run["cov"], strict=True):
        values, vectors = np.linalg.eigh(cov)
        ellipses.append(mu + unit @ (vectors * np.sqrt(values)).T * 2)
    return vm.scientific_canvas(
        points=x[:300], colors=run["responsibility"][:300] @ colors, paths=ellipses
    )


def gaussian_mixture():
    def build():
        data = mnist_projection(600, 300, tuple(range(8)))
        x = data["train"]
        a, b = toy.gmm_em(x, 8, 60, SEED), toy.gmm_em(x, 8, 60, 14)
        return dict(
            **projection_arrays(data),
            **parameter_arrays(a, "a_"),
            **parameter_arrays(b, "b_"),
            likelihoods=[
                gmm_log_likelihood(x, a),
                gmm_log_likelihood(data["test"], a),
                gmm_log_likelihood(x, b),
                gmm_log_likelihood(data["test"], b),
            ],
        )

    d = cached(
        "gmm_mnist",
        build,
        train=600,
        test=300,
        digits=list(range(8)),
        components=8,
        steps=60,
        initialization_seeds=[51, 14],
    )
    a = {k[2:]: v for k, v in d.items() if k.startswith("a_")}
    b = {k[2:]: v for k, v in d.items() if k.startswith("b_")}
    x = d["data_train"]
    return result(
        "gaussian-mixture-em",
        "Eight-component EM fitted to real MNIST training-only PCA coordinates",
        [
            points(
                "Measured digits, not generated Gaussian blobs",
                "Colours show real labels for context; EM does not receive those labels.",
                x[:300],
                d["data_train_labels"][:300],
            ),
            images(
                "Learned responsibilities and covariance contours",
                "Components are density components, not guaranteed digit classes.",
                [mixture_picture(x, a), mixture_picture(x, b)],
                ["seed 51", "seed 14"],
            ),
            curve(
                "Log likelihood during actual EM updates",
                "Both initializations fit the same measured training points.",
                [a["loss"], b["loss"]],
                ["seed 51", "seed 14"],
            ),
            bars(
                "Evaluate fitted densities on unseen images",
                "Mean log density uses the same fixed training-only coordinate map.",
                d["likelihoods"],
                ["seed51 train", "seed51 test", "seed14 train", "seed14 test"],
            ),
        ],
        seeds=[51, 14],
    )


def curse_dimensions():
    def build():
        import math

        data = mnist_projection(1000, 200, dimensions=49)
        dims = np.array([2, 4, 8, 16, 32, 49])
        ratio = []
        accuracy = []
        for dim in dims:
            distances = np.linalg.norm(
                data["test"][:, None, :dim] - data["train"][None, :, :dim], axis=-1
            )
            ratio.append(np.mean(distances.min(1) / np.maximum(distances.max(1), 1e-12)))
            accuracy.append(
                np.mean(data["train_labels"][distances.argmin(1)] == data["test_labels"])
            )
        volume = [
            np.exp(dim / 2 * np.log(np.pi) - math.lgamma(dim / 2 + 1) - dim * np.log(2))
            for dim in dims
        ]
        return dict(
            dims=dims, ratio=ratio, accuracy=accuracy, volume=volume, **projection_arrays(data)
        )

    d = cached("dimension_mnist", build, train=1000, test=200, dimensions=[2, 4, 8, 16, 32, 49])
    return result(
        "curse-of-dimensionality",
        "Measured MNIST neighbourhoods as more training-only PCA coordinates are retained",
        [
            curve(
                "Measure distance concentration",
                "Mean nearest/farthest ratio for 200 test images against 1,000 training images.",
                d["ratio"],
                ["distance ratio"],
                d["dims"],
            ),
            curve(
                "Separate analytic geometry reference",
                "Ball/cube volume is an exact formula, not a measured MNIST learning curve.",
                d["volume"],
                ["analytic ball/cube"],
                d["dims"],
            ),
            curve(
                "More dimensions can also retain useful signal",
                "Actual nearest-neighbour test accuracy; no degradation is imposed.",
                d["accuracy"],
                ["test 1-NN accuracy"],
                d["dims"],
            ),
            bars(
                "The result depends on the data and metric",
                "These full-coordinate measurements describe this MNIST subset.",
                [d["ratio"][-1], d["accuracy"][-1]],
                ["distance ratio", "test accuracy"],
            ),
        ],
        seed=SEED,
        scope="Measured MNIST analysis plus explicitly analytic volume reference",
    )


def tsne_affinity(x, perplexity):
    """Symmetric t-SNE affinities with per-row entropy-matched bandwidths."""
    dist = ((x[:, None] - x[None]) ** 2).sum(-1)
    n, p = len(x), np.zeros_like(dist)
    for i in range(n):
        mask = np.arange(n) != i
        beta, lo, hi = 1.0, 0.0, np.inf
        for _ in range(50):
            values = -dist[i, mask] * beta
            values -= values.max()
            row = np.exp(values)
            row /= row.sum()
            entropy = -np.sum(row * np.log(row + 1e-300))
            if entropy > np.log(perplexity):
                lo = beta
                beta = beta * 2 if np.isinf(hi) else (beta + hi) / 2
            else:
                hi = beta
                beta = (beta + lo) / 2
        p[i, mask] = row
    return (p + p.T) / (2 * n)


def tsne_fit(x, perplexity, steps=400, seed=SEED):
    rng = np.random.default_rng(seed)
    p = tsne_affinity(x, perplexity)
    z = rng.normal(0, 1e-3, (len(x), 2))
    velocity = np.zeros_like(z)
    losses = []
    for step in range(steps):
        dif = z[:, None] - z[None]
        inv = 1 / (1 + (dif**2).sum(-1))
        np.fill_diagonal(inv, 0)
        q = inv / inv.sum()
        pp = p * (4 if step < 80 else 1)
        gradient = 4 * np.sum(((pp - q) * inv)[:, :, None] * dif, axis=1)
        velocity = 0.7 * velocity - 50 * gradient
        z += velocity
        z -= z.mean(0)
        losses.append(np.sum(p * np.log((p + 1e-300) / (q + 1e-300))))
    return z, np.array(losses)


def tsne_umap():
    def build():
        _, _, x, y = mnist_data()
        x = vm.downsample(x[:240])
        maps, losses = [], []
        for perplexity in (5, 30, 100):
            z, loss = tsne_fit(x, perplexity)
            maps.append(z)
            losses.append(loss)
        return dict(
            maps=maps,
            losses=losses,
            labels=y[:240],
            fit_indices=np.arange(240),
            fit_pixels=x,
            fit_split=np.array("MNIST test split used as embedding fit data"),
        )

    d = cached("tsne", build, samples=240, perplexities=[5, 30, 100], steps=400, resolution=7)
    return result(
        "tsne-and-umap",
        "Exact t-SNE on 240 downsampled MNIST examples",
        [
            points(
                "Perplexity five",
                "Local affinities are fitted to the same 240 examples in every run.",
                d["maps"][0],
                d["labels"],
            ),
            points(
                "Perplexity thirty",
                "A larger perplexity changes which neighbourhoods the map preserves.",
                d["maps"][1],
                d["labels"],
            ),
            points(
                "Perplexity one hundred",
                "Cluster spacing in these maps is not a calibrated original-space distance.",
                d["maps"][2],
                d["labels"],
            ),
            curve(
                "Optimisation objectives differ",
                "Perplexity changes target affinities; KL values are not quality rankings.",
                d["losses"],
                ["5", "30", "100"],
            ),
        ],
        scope="t-SNE only, matching TODO body; UMAP is not implemented",
    )


def roc_calibration():
    from . import mnist

    p, _ = mnist.train()
    _, _, x, y = mnist_data()
    prob = mnist._forward(p, x[:2000])[2][:, 7]
    target = y[:2000] == 7
    thresholds = np.r_[np.inf, np.unique(prob)[::-1], -np.inf]
    tp = np.array([np.sum(target & (prob >= t)) for t in thresholds])
    fp = np.array([np.sum(~target & (prob >= t)) for t in thresholds])
    tpr, fpr = tp / target.sum(), fp / (~target).sum()
    precision = tp / np.maximum(tp + fp, 1)
    auc = np.trapezoid(tpr, fpr)
    bins = np.linspace(0, 1, 11)
    confidence, frequency = [], []
    for a, b in itertools.pairwise(bins):
        selected = (prob >= a) & (prob <= b if b == 1 else prob < b)
        if selected.any():
            confidence.append(prob[selected].mean())
            frequency.append(target[selected].mean())
    at = prob >= 0.5
    confusion = np.array(
        [[np.sum(~target & ~at), np.sum(~target & at)], [np.sum(target & ~at), np.sum(target & at)]]
    )
    return result(
        "roc-and-calibration",
        "Digit seven versus all others on 2,000 held-out MNIST examples",
        [
            matrix(
                "The confusion matrix at threshold 0.5",
                "Rows are actual classes; columns are predicted classes (negative, then positive).",
                confusion,
            ),
            curve(
                "Every possible decision threshold",
                "True-positive rate is traced against false-positive rate.",
                tpr,
                ["ROC"],
                fpr,
                ("AUC %.4f" % auc,),
            ),
            curve(
                "Precision and recall",
                "Precision reflects this test subset's measured class prevalence.",
                precision,
                ["precision"],
                tpr,
                ("positive prevalence %.1f%%" % (100 * target.mean()),),
            ),
            curve(
                "Probability calibration",
                "Within each occupied confidence bin, measure the fraction of sevens.",
                [frequency, confidence],
                ["observed", "perfect"],
                confidence,
            ),
        ],
    )


_BUILDERS = dict(
    zip(
        NAMES,
        [
            receptive_field,
            vit_attention,
            clip_contrastive,
            adversarial_examples,
            object_detection,
            segmentation,
            batchnorm,
            style_transfer,
            gan_training,
            normalizing_flows,
            flow_matching,
            classifier_free_guidance,
            ddpm_ddim,
            vae_collapse,
            vq_vae,
            autoregressive_pixels,
            bias_variance,
            cross_validation,
            svm_kernels,
            gradient_boosting,
            gaussian_mixture,
            curse_dimensions,
            tsne_umap,
            roc_calibration,
        ],
        strict=True,
    )
)


@cache
def make(name):
    """Compute a four-phase measured study for one proposed animation."""
    return _BUILDERS[name]()
