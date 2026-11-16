"""Verify a completed local TODO review gallery without rendering or publishing.

    python3 tools/verify_next_review.py --out build/review-realdata-100

The default scope is all 100 catalog entries. --start/--end verify a selected
range and permit other catalog entries in the same gallery. Every selected MP4
is independently probed and completely decoded. A JSON report is written to
the gallery, and any failed check produces a nonzero exit status.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from voxanim.next_catalog import ITEMS  # noqa: E402


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def array_digest(value):
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def local_path(out, relative):
    require(isinstance(relative, str), "Expected a relative gallery path")
    require(not Path(relative).is_absolute(), "Absolute gallery reference: " + relative)
    path = (out / relative).resolve()
    require(path.is_relative_to(out.resolve()), "Reference leaves gallery: " + relative)
    require(path.is_file(), "Missing gallery file: " + relative)
    return path


def read_arrays(path):
    """Loading every member catches object/pickle payloads and corrupt ZIP data."""
    with np.load(path, allow_pickle=False) as archive:
        require(len(archive.files) == len(set(archive.files)), "Duplicate NPZ member")
        arrays = {}
        for key in archive.files:
            value = archive[key]
            require(not value.dtype.hasobject, "Object array: " + key)
            if value.dtype.kind in "buifc":
                require(np.isfinite(value).all(), "Non-finite NPZ array: " + key)
            arrays[key] = value
    return arrays


def array_references(value):
    if isinstance(value, dict):
        if set(value) == {"array"}:
            yield value["array"]
        else:
            for child in value.values():
                yield from array_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from array_references(child)


def artifact_references(value):
    """Match the two metadata artifact forms exported by review_next.py."""
    if isinstance(value, dict):
        artifact = value.get("artifact")
        if isinstance(artifact, str):
            yield artifact, value.get("artifact_sha256")
        elif isinstance(artifact, dict) and "path" in artifact:
            yield artifact["path"], artifact.get("sha256")
        for child in value.values():
            yield from artifact_references(child)
    elif isinstance(value, list):
        for child in value:
            yield from artifact_references(child)


def verify_artifact(out, item):
    path = local_path(out, item["path"])
    actual = digest(path)
    require(actual == item["sha256"], "Run artifact SHA mismatch: " + item["path"])
    arrays = read_arrays(path)
    result = dict(path=item["path"], sha256=actual, arrays=len(arrays), receipt=None)
    if "receipt" not in item:
        require("receipt_sha256" not in item, "Receipt hash has no receipt path")
        return result
    receipt_path = local_path(out, item["receipt"])
    require(
        digest(receipt_path) == item.get("receipt_sha256"),
        "Run receipt SHA mismatch: " + item["receipt"],
    )
    receipt = json.loads(receipt_path.read_text())
    require(receipt["artifact_sha256"] == actual, "Receipt identifies a different artifact")
    require(Path(receipt["artifact"]).name == item["name"], "Receipt artifact name mismatch")
    descriptions = receipt["arrays"]
    require(set(descriptions) == set(arrays), "Receipt/NPZ array inventory mismatch")
    for key, value in arrays.items():
        expected = descriptions[key]
        require(list(value.shape) == expected["shape"], "Receipt shape mismatch: " + key)
        require(str(value.dtype) == expected["dtype"], "Receipt dtype mismatch: " + key)
        require(array_digest(value) == expected["sha256"], "Receipt array SHA mismatch: " + key)
    identity = {
        k: v for k, v in receipt.items() if k not in {"run_id", "artifact", "artifact_sha256"}
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    require(run_id == receipt["run_id"], "Run identity does not match receipt contents")
    result.update(receipt=item["receipt"], run_id=run_id)
    return result


def verify_video(path, manifest):
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        "stream=codec_name,width,height,nb_frames,nb_read_frames,avg_frame_rate,duration",
        "-of",
        "json",
        str(path),
    ]
    probe = subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    streams = json.loads(probe.stdout)["streams"]
    require(len(streams) == 1, "Expected one selected video stream")
    stream = streams[0]
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    expected_fps = float(manifest["fps"])
    frames = int(manifest["frames"])
    require(expected_fps > 0 and frames > 0, "Non-positive frame rate or frame count")
    require(abs(fps - expected_fps) < 1e-9, "Video frame rate differs from manifest")
    require(int(stream["nb_read_frames"]) == frames, "Decoded frame count differs from manifest")
    require(int(stream["nb_frames"]) == frames, "Container frame count differs from manifest")
    require(int(stream["width"]) == manifest["width"], "Video width differs from manifest")
    require(int(stream["height"]) == manifest["height"], "Video height differs from manifest")
    duration = float(stream["duration"])
    require(math.isfinite(duration), "Non-finite video duration")
    require(abs(duration - frames / fps) <= 0.02, "Video duration differs from frame timeline")
    require(math.ceil(float(manifest["duration"]) * fps) == frames, "Manifest timeline mismatch")
    stored = manifest.get("stream", {})
    require(abs(float(stored["duration"]) - duration) < 1e-6, "Stored stream duration differs")
    for key in ("width", "height", "nb_frames", "codec_name", "avg_frame_rate"):
        require(str(stream[key]) == str(stored.get(key)), "Stored ffprobe field differs: " + key)
    decode = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    require(decode.returncode == 0, "Full video decode failed: " + decode.stderr[-2000:])
    require(not decode.stderr.strip(), "Decoder reported errors: " + decode.stderr[-2000:])
    return dict(stream=stream, full_decode=True)


def verify_one(out, item):
    name = item["preset"]
    result = dict(number=item["number"], preset=name, ok=False)
    try:
        manifest_path = local_path(out, "media/" + name + ".json")
        manifest = json.loads(manifest_path.read_text())
        result["manifest_sha256"] = digest(manifest_path)
        require(manifest["preset"] == name, "Manifest preset mismatch")
        require(manifest["number"] == item["number"], "Manifest catalog number mismatch")
        require(
            manifest.get("published") is False, "Manifest does not identify an unpublished review"
        )
        require(manifest.get("layout_qa_issues") == 0, "Manifest records layout QA issues")
        video = local_path(out, "media/" + name + ".mp4")
        require(digest(video) == manifest["sha256"], "Video SHA differs from manifest")
        expected_sidecars = {
            "data/" + name + ".json",
            "data/" + name + ".npz",
            "media/" + name + ".jpg",
        }
        require(set(manifest["sidecar_sha256"]) == expected_sidecars, "Sidecar inventory mismatch")
        for relative, expected in manifest["sidecar_sha256"].items():
            require(
                digest(local_path(out, relative)) == expected, "Sidecar SHA mismatch: " + relative
            )
        info = json.loads(local_path(out, "data/" + name + ".json").read_text())
        require(
            info["metadata"] == manifest["study_metadata"], "Study metadata differs from manifest"
        )
        require(
            info.get("run_artifacts", []) == manifest.get("run_artifacts", []),
            "Run references differ",
        )
        require(len(info["panels"]) == 4, "Expected four study panels")
        arrays = read_arrays(local_path(out, "data/" + name + ".npz"))
        require(
            set(array_references(info["panels"])) == set(arrays),
            "Plot array references differ from NPZ",
        )
        artifacts = info.get("run_artifacts", [])
        references = {(Path(p).name, h) for p, h in artifact_references(info["metadata"])}
        copied = {(a["name"], a["sha256"]) for a in artifacts}
        require(references == copied, "Metadata run artifacts do not match copied evidence")
        result["artifacts"] = [verify_artifact(out, a) for a in artifacts]
        result["video"] = verify_video(video, manifest)
        result["plot_arrays"] = len(arrays)
        result["evidence_kind"] = info["metadata"].get("evidence_kind", "unspecified")
        result["panel_evidence"] = info["metadata"].get("panel_evidence", [])
        result["evidence_declarations"] = {
            key: info["metadata"][key]
            for key in ("source", "method", "scope", "evidence_types")
            if key in info["metadata"]
        }
        result["ok"] = True
    except Exception as error:
        result["error"] = str(error)
    return result


def verify_inventory(out, selected):
    wanted = {i["preset"] for i in selected}
    known = {i["preset"] for i in ITEMS}
    errors, counts = [], {}
    for directory, suffix in (
        ("media", ".mp4"),
        ("media", ".json"),
        ("media", ".jpg"),
        ("data", ".json"),
        ("data", ".npz"),
    ):
        present = {p.name.removesuffix(suffix) for p in (out / directory).glob("*" + suffix)}
        missing, unknown = wanted - present, present - known
        # An incomplete selected video is always a failure; completed entries
        # outside a requested subset are deliberately permitted.
        unfinished = {p for p in unknown if p.removesuffix(".part") in wanted}
        if missing:
            errors.append(f"Missing {directory}/*{suffix}: {', '.join(sorted(missing))}")
        if unknown and len(selected) == len(ITEMS):
            errors.append(f"Unexpected {directory}/*{suffix}: {', '.join(sorted(unknown))}")
        elif unfinished:
            errors.append(f"Incomplete selected videos: {', '.join(sorted(unfinished))}")
        counts[directory + "/*" + suffix] = dict(
            expected=len(wanted), selected=len(present & wanted), total=len(present)
        )
    return errors, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "review-realdata-100")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=100)
    parser.add_argument(
        "--jobs", type=int, default=4, help="Parallel complete video decodes (default: 4)"
    )
    args = parser.parse_args(argv)
    if not 1 <= args.start <= args.end <= 100 or args.jobs < 1:
        parser.error("Require 1 <= start <= end <= 100 and jobs >= 1")
    out = args.out.resolve()
    if not out.is_dir():
        parser.error("Gallery directory does not exist: " + str(out))
    selected = [item for item in ITEMS if args.start <= item["number"] <= args.end]
    if sorted(item["number"] for item in selected) != list(range(args.start, args.end + 1)) or len(
        {item["preset"] for item in selected}
    ) != len(selected):
        parser.error("Catalog does not contain exactly one preset for every requested number")
    errors, inventory = verify_inventory(out, selected)
    results = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(verify_one, out, item): item for item in selected}
        for future in as_completed(pending):
            result = future.result()
            results.append(result)
            status = "PASS" if result["ok"] else "FAIL " + result["error"]
            print(
                f"{result['number']:3} {result['preset']}: {status}",
                flush=True,
            )
    results.sort(key=lambda row: row["number"])
    kinds = Counter(str(row.get("evidence_kind", "unverified")) for row in results)
    report = dict(
        ok=not errors and all(row["ok"] for row in results),
        verified_at=datetime.now(UTC).isoformat(),
        gallery=str(out),
        start=args.start,
        end=args.end,
        requested=len(selected),
        passed=sum(row["ok"] for row in results),
        inventory=inventory,
        inventory_errors=errors,
        evidence_kinds=dict(kinds),
        results=results,
        scope=(
            "File integrity, copied evidence consistency, finite arrays and video decoding. "
            "Does not establish scientific validity or visual readability; "
            "analytic and fixed-model studies need not have training receipts."
        ),
    )
    report_path = out / "verification-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    status = "PASS" if report["ok"] else "FAIL"
    print(f"{status}: {report['passed']}/{len(selected)} entries; report {report_path}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
