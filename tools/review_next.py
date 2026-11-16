"""Render the TODO study set to a local-only review gallery; never publish.

Run with the environment's Python (NumPy and Pillow), plus ffmpeg/ffprobe:
    python3 tools/review_next.py --jobs 8 --fps 24
Resume is automatic when the source/study fingerprint and video hashes match.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Keep each independent study from spawning another full machine of BLAS threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from voxanim import presets  # noqa: E402
from voxanim.next_catalog import ITEMS, build_study  # noqa: E402
from voxanim.qa import check  # noqa: E402
from voxanim.raster import to_uint8  # noqa: E402
from voxanim.scene import Renderer, Style  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def export_run_artifacts(metadata, out):
    """Copy referenced, checksum-verified run artifacts into the local review."""
    references = {}

    def visit(value):
        if isinstance(value, dict):
            artifact = value.get("artifact")
            if isinstance(artifact, str):
                references[artifact] = value.get("artifact_sha256")
            elif isinstance(artifact, dict) and "path" in artifact:
                references[artifact["path"]] = artifact.get("sha256")
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(metadata)
    directory = out / "evidence"
    directory.mkdir(parents=True, exist_ok=True)
    exported = []
    for source, expected in sorted(references.items()):
        path = Path(source).resolve()
        if not path.is_relative_to(ROOT / "build"):
            raise ValueError("Run artifact is outside the repository build directory: " + source)
        actual = digest(path)
        if not expected or actual != expected:
            raise ValueError("Run artifact checksum mismatch: " + source)
        destination = directory / (actual + path.suffix)
        if not destination.exists() or digest(destination) != actual:
            shutil.copyfile(path, destination)
        item = dict(name=path.name, path="evidence/" + destination.name, sha256=actual)
        receipt = path.with_suffix(".json")
        if receipt.exists():
            receipt_hash = digest(receipt)
            receipt_dest = directory / (receipt_hash + ".json")
            if not receipt_dest.exists():
                shutil.copyfile(receipt, receipt_dest)
            item.update(receipt="evidence/" + receipt_dest.name, receipt_sha256=receipt_hash)
        exported.append(item)
    return exported


def payload(study):
    arrays, panels = {}, []
    for i, p in enumerate(study.panels):
        data = {}
        for key, value in p.data.items():
            if isinstance(value, np.ndarray):
                name = "%d_%s" % (i, key)
                arrays[name] = value
                data[key] = {"array": name}
            else:
                data[key] = value
        panels.append(dict(title=p.title, caption=p.caption, kind=p.kind, notes=p.notes, data=data))
    info = dict(title=study.title, subtitle=study.subtitle, metadata=study.metadata, panels=panels)
    encoded = json.dumps(info, ensure_ascii=False, sort_keys=True, default=str).encode()
    h = hashlib.sha256(encoded)
    for key, value in sorted(arrays.items()):
        h.update(key.encode())
        h.update(str(value.shape).encode())
        h.update(str(value.dtype).encode())
        h.update(value.tobytes())
    for name in (
        "study.py",
        "mech.py",
        "diagram.py",
        "scene.py",
        "netanim.py",
        "layout.py",
        "camera.py",
        "color.py",
        "easing.py",
        "fonts.py",
        "raster.py",
    ):
        h.update((ROOT / "voxanim" / name).read_bytes())
    # what draws the studies: a change to any of the drawing modules re-renders
    for path in sorted((ROOT / "voxanim").glob("mech_*.py")):
        h.update(path.read_bytes())
    h.update(Path(__file__).read_bytes())
    return info, arrays, h.hexdigest()


def render_one(job):
    item, out, fps, width, qa_samples = job
    out = Path(out)
    name = item["preset"]
    study = build_study(item["module"], item["name"])
    info, arrays, fingerprint = payload(study)
    info["run_artifacts"] = export_run_artifacts(study.metadata, out)
    base = out / "media" / name
    video, manifest = base.with_suffix(".mp4"), base.with_suffix(".json")
    fingerprint += "-%d-%d" % (fps, width)
    if video.exists() and manifest.exists():
        old = json.loads(manifest.read_text())
        sidecars = [out / "data" / (name + ext) for ext in (".json", ".npz")]
        sidecars.append(base.with_suffix(".jpg"))
        if (
            old.get("fingerprint") == fingerprint
            and old["sha256"] == digest(video)
            and old.get("layout_qa_samples", 0) >= qa_samples
            and all(
                p.exists()
                and old.get("sidecar_sha256", {}).get(str(p.relative_to(out))) == digest(p)
                for p in sidecars
            )
        ):
            return name, "cached", old
    issues = check(name, samples=qa_samples, verbose=False)
    if issues:
        check(name, samples=qa_samples, verbose=True)
        raise ValueError("%s has %d layout QA issues" % (name, issues))
    anim = presets.build(name)
    # the transcript: each phase as the film names it, with the subtitle it shows
    info["phases"] = [dict(name=phase, subtitle=anim.subtitle_for(k), seconds=seconds)
                      for k, (phase, seconds) in enumerate(anim.PHASES)]
    style = Style(width=width, height=width * 9 // 16, ss=1, motion_blur=False, reflection=0.0)
    renderer = Renderer(anim.cam, style)
    total = int(np.ceil(anim.duration * fps))
    temp = base.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "%dx%d" % (style.width, style.height),
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "22",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        "1",
        "-fflags",
        "+bitexact",
        "-flags:v",
        "+bitexact",
        "-map_metadata",
        "-1",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    start = time.monotonic()
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        try:
            for frame in range(total):
                rgb = to_uint8(renderer.render_at(anim.scene, frame / fps, 1 / fps))
                process.stdin.write(rgb.tobytes())
            process.stdin.close()
            error = process.stderr.read()
            result = process.wait()
            if result:
                raise RuntimeError(error.decode())
        except BaseException:
            process.kill()
            process.wait()
            raise
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames,avg_frame_rate,duration",
                "-of",
                "json",
                str(temp),
            ]
        )
    )
    stream = probe["streams"][0]
    if int(stream["nb_frames"]) != total or abs(float(stream["duration"]) - total / fps) > 0.02:
        raise ValueError("Encoded stream length does not match rendered timeline")
    temp.replace(video)
    (out / "data" / (name + ".json")).write_text(
        json.dumps(info, indent=2, ensure_ascii=False, default=str) + "\n"
    )
    np.savez_compressed(out / "data" / (name + ".npz"), **arrays)
    # Four frames, each well into a phase, make a browsable poster without video playback.
    sheet = Image.new("RGB", (width, width * 9 // 16))
    spans = [anim.spans[round(i * (len(anim.spans) - 1) / 3)] for i in range(4)]
    for i, second in enumerate(a + 0.6 * (b - a) for a, b in spans):
        image = Image.fromarray(to_uint8(renderer.render(anim.scene(second))))
        image = image.resize((width // 2, style.height // 2), Image.Resampling.LANCZOS)
        sheet.paste(image, ((i % 2) * width // 2, (i // 2) * style.height // 2))
    sheet.save(base.with_suffix(".jpg"), quality=88)
    result = dict(
        preset=name,
        number=item["number"],
        fingerprint=fingerprint,
        sha256=digest(video),
        frames=total,
        fps=fps,
        duration=anim.duration,
        width=style.width,
        height=style.height,
        quality="local review, 1x supersampling",
        layout_qa_samples=qa_samples,
        layout_qa_issues=0,
        stream=stream,
        elapsed_seconds=round(time.monotonic() - start, 2),
        study_metadata=study.metadata,
        run_artifacts=info["run_artifacts"],
        sidecar_sha256={
            str(path.relative_to(out)): digest(path)
            for path in (
                out / "data" / (name + ".json"),
                out / "data" / (name + ".npz"),
                base.with_suffix(".jpg"),
            )
        },
        published=False,
    )
    manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n")
    return name, "rendered", result


def write_gallery(out):
    cards = []
    for item in ITEMS:
        name = item["preset"]
        manifest = out / "media" / (name + ".json")
        if not manifest.exists():
            continue
        info = json.loads((out / "data" / (name + ".json")).read_text())
        esc = html.escape
        transcript = "".join(
            "<li><b>%s</b> — %s</li>" % (esc(p["name"]), esc(p["subtitle"])) for p in info["phases"]
        )
        limits = info["metadata"].get("limitations", [])
        scope = (
            ""
            if not limits
            else "<p><b>Study limits:</b> "
            + "; ".join(esc(str(value)) for value in limits)
            + "</p>"
        )
        artifacts = info.get("run_artifacts", [])
        evidence = "".join(
            '<li><a href="%s">Run arrays / checkpoint: %s</a>%s</li>'
            % (
                esc(a["path"]),
                esc(a["name"]),
                (' · <a href="%s">Run receipt</a>' % esc(a["receipt"])) if "receipt" in a else "",
            )
            for a in artifacts
        )
        evidence = (
            "<p><b>Recorded run evidence</b></p><ul>" + evidence + "</ul>"
            if evidence
            else "<p>No new training checkpoint is attached to this "
            "calculation or fixed-model analysis; inspect its stated method.</p>"
        )
        cards.append(
            """<article data-search="{search}"><h2>{number}. {title}</h2>
<p class="family">{family} · {preset}</p>
<video controls loop playsinline preload="none"
poster="media/{preset}.jpg" src="media/{preset}.mp4"></video>
<p>{subtitle}</p><details><summary>Read the four phases and inspect measurements</summary>
<ol>{transcript}</ol>{scope}{evidence}
<a href="data/{preset}.json">Study details</a> ·
<a href="data/{preset}.npz">Plot arrays</a> ·
<a href="media/{preset}.json">Render verification</a> ·
<a href="media/{preset}.mp4" download>Download video</a></details>
<label>Review notes<textarea data-key="{preset}"
placeholder="Notes stay in this browser only"></textarea></label></article>""".format(
                search=esc((item["section"] + " " + name + " " + info["title"]).lower()),
                number=item["number"],
                title=esc(info["title"]),
                family=esc(item["section"]),
                preset=esc(name),
                subtitle=esc(info["subtitle"]),
                transcript=transcript,
                scope=scope,
                evidence=evidence,
            )
        )
    page = (
        Path(__file__)
        .with_name("review_next_template.html")
        .read_text()
        .replace("COUNT", str(len(cards)))
        .replace("CARDS", "\n".join(cards))
    )
    (out / "index.html").write_text(page)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "review-next-100")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=100)
    ap.add_argument("--qa-samples", type=int, default=120)
    ap.add_argument("--gallery-only", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if args.gallery_only:
        write_gallery(out)
        return
    for directory in ("media", "data"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    jobs = [
        (item, str(out), args.fps, args.width, args.qa_samples)
        for item in ITEMS
        if args.start <= item["number"] <= args.end
    ]
    errors = []
    with ProcessPoolExecutor(args.jobs, mp_context=mp.get_context("spawn")) as ex:
        pending = {ex.submit(render_one, job): job[0] for job in jobs}
        for future in as_completed(pending):
            item = pending[future]
            try:
                name, status, result = future.result()
                print(
                    "%3d %-44s %s (%.1fs)"
                    % (item["number"], name, status, result["elapsed_seconds"]),
                    flush=True,
                )
                write_gallery(out)
            except Exception as error:
                errors.append(dict(preset=item["preset"], error=repr(error)))
                print("FAILED", item["preset"], repr(error), flush=True)
    (out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
    write_gallery(out)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
