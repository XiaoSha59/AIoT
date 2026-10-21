"""Isolated PyTorch/ORT CPU benchmarks on the same device and real test inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
from PIL import Image

from edgeface.preprocess import PREPROCESS, reject, save_json, softmax, tensor_from_rgb


def summary(values: list[float]) -> dict:
    if not values:
        raise ValueError("No measurements")
    return {"samples": len(values), "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95)),
            "std_ms": float(np.std(values)), "min_ms": min(values), "max_ms": max(values)}


def hardware() -> dict:
    def number(path: str, divisor: float = 1):
        try:
            return float(Path(path).read_text())/divisor
        except (OSError, ValueError):
            return None
    try:
        throttled = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True,
                                   text=True, timeout=2).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        throttled = None
    return {"temperature_c": number("/sys/class/thermal/thermal_zone0/temp", 1000),
            "cpu_frequency_mhz": number("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000),
            "throttled": throttled, "system_available_ram_bytes": psutil.virtual_memory().available}


class MemorySampler:
    """Sample RSS during a phase; Linux ru_maxrss additionally covers short-lived peaks."""
    def __init__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self.stop.wait(.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)


def package_footprint(names: list[str]) -> dict:
    """Installed files owned by these distributions, excluding unrelated/shared dependencies."""
    seen = set()
    logical = allocated = files = 0
    versions = {}
    for name in names:
        dist = importlib.metadata.distribution(name)
        versions[name] = dist.version
        for entry in dist.files or []:
            path = Path(os.path.abspath(dist.locate_file(entry)))
            if path in seen:
                continue
            seen.add(path)
            try:
                info = path.lstat()
            except OSError:
                continue
            if stat.S_ISREG(info.st_mode):
                files += 1
                logical += info.st_size
                allocated += info.st_blocks*512
    return {"packages": versions, "logical_bytes": logical, "allocated_bytes": allocated,
            "files": files, "definition": "Distribution-owned installed files; excludes common/transitive dependencies and OS"}


def classification_metrics(logits: np.ndarray, labels: np.ndarray, classes: list[str], threshold: float) -> dict:
    probs = softmax(logits)
    guesses = np.array([classes.index(reject(p, classes, threshold)[0]) for p in probs])
    matrix = np.zeros((len(classes), len(classes)), dtype=int)
    for truth, guess in zip(labels, guesses):
        matrix[truth, guess] += 1
    shifted = np.asarray(logits, dtype=np.float64)-np.max(logits, axis=1, keepdims=True)
    loss = np.log(np.exp(shifted).sum(axis=1))-shifted[np.arange(len(labels)), labels]
    per_class = {}
    for i, name in enumerate(classes):
        tp, count, predicted = int(matrix[i, i]), int(matrix[i].sum()), int(matrix[:, i].sum())
        precision, recall = tp/predicted if predicted else 0, tp/count if count else 0
        per_class[name] = {"precision": precision, "recall": recall,
                           "f1": 2*precision*recall/(precision+recall) if precision+recall else 0,
                           "support": count}
    unknown = classes.index("Unknown")
    known_mask = labels != unknown
    unknown_mask = ~known_mask
    if not known_mask.any() or not unknown_mask.any():
        raise ValueError("Quality evaluation requires known and Unknown inputs")
    return {"samples": len(labels), "accuracy": float(np.mean(guesses == labels)),
            "test_cross_entropy_unweighted": float(loss.mean()),
            "macro_f1": float(np.mean([v["f1"] for v in per_class.values()])),
            "known_member_accuracy": float(np.mean(guesses[known_mask] == labels[known_mask])),
            "unknown_rejection_rate": float(np.mean(guesses[unknown_mask] == unknown)),
            "false_acceptance_rate": float(np.mean(guesses[unknown_mask] != unknown)),
            "per_class": per_class, "confusion_matrix": matrix.tolist()}


def read_tensor(data: Path, row: dict) -> np.ndarray:
    path = data/row["file"]
    if not path.resolve().is_relative_to(data.resolve()):
        raise ValueError("Input escapes dataset")
    if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
        raise ValueError("Real input checksum mismatch")
    with Image.open(path) as image:
        return tensor_from_rgb(image)


def worker(args) -> None:
    manifest_path = args.data/"manifest.json"
    manifest = json.loads(manifest_path.read_text())
    calibration = json.loads((args.model/"threshold.json").read_text())
    export = json.loads((args.model/"export_validation.json").read_text())
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if calibration["manifest_sha256"] != manifest_hash or calibration["source"] != "validation":
        raise ValueError("Benchmark dataset/calibration provenance mismatch")
    classes = json.loads((args.model/"classes.json").read_text())
    if classes != manifest["classes"] or json.loads((args.model/"preprocessing.json").read_text()) != PREPROCESS:
        raise ValueError("Class/preprocessing mismatch")
    rows = [r for r in manifest["samples"] if r["split"] == "test"]
    # Same three real input tensors in every worker, <2 MiB total. Never preload 181 images.
    selected = [rows[i*len(rows)//3] for i in range(3)]
    inputs = [read_tensor(args.data, row) for row in selected]
    process = psutil.Process()
    start_hardware = hardware()
    rss_before = process.memory_info().rss
    load_start = time.perf_counter()
    with MemorySampler() as loading:
        if args.engine == "pytorch":
            import torch
            from training.model import load_checkpoint
            torch.set_num_threads(args.threads)
            torch.set_num_interop_threads(1)
            digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            if digest != export["checkpoint_sha256"]:
                raise ValueError("PyTorch weights differ from ONNX source weights")
            model, metadata = load_checkpoint(args.checkpoint, "cpu")
            if metadata["manifest_sha256"] != manifest_hash or metadata["classes"] != classes:
                raise ValueError("Checkpoint provenance mismatch")
            tensors = [torch.from_numpy(x) for x in inputs]
            def forward(index):
                return model(tensors[index])
            def evaluate(tensor):
                return model(torch.from_numpy(tensor)).numpy()[0]
            forward = torch.inference_mode()(forward)
            evaluate = torch.inference_mode()(evaluate)
            packages = ["torch", "torchvision"]
            model_file = args.checkpoint
        else:
            import onnxruntime as ort
            model_file = args.model/"face_classifier.onnx"
            if hashlib.sha256(model_file.read_bytes()).hexdigest() != export["onnx_sha256"]:
                raise ValueError("ONNX digest mismatch")
            options = ort.SessionOptions()
            options.intra_op_num_threads = args.threads
            options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(str(model_file), options, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            if session.get_inputs()[0].shape != [1, 3, 224, 224]:
                raise ValueError("Expected fixed batch-one model")
            def forward(index):
                return session.run(None, {input_name: inputs[index]})[0]
            def evaluate(tensor):
                return session.run(None, {input_name: tensor})[0][0]
            packages = ["onnxruntime"]
    startup_ms = (time.perf_counter()-load_start)*1000
    rss_loaded = process.memory_info().rss
    for i in range(args.warmup):
        forward(i % len(inputs))
    rss_warm = process.memory_info().rss
    latency = []
    cpu_start = process.cpu_times()
    started = time.perf_counter()
    with MemorySampler() as inference:
        for i in range(args.iterations):
            tick = time.perf_counter()
            forward(i % len(inputs))
            latency.append((time.perf_counter()-tick)*1000)
    elapsed = time.perf_counter()-started
    cpu_end = process.cpu_times()
    cpu_s = (cpu_end.user+cpu_end.system)-(cpu_start.user+cpu_start.system)
    result = {"engine": args.engine, "device": platform.node(), "architecture": platform.machine(),
              "os": platform.platform(), "cpu_cores": psutil.cpu_count(), "threads": args.threads,
              "inter_op_threads": 1, "precision": "FP32", "batch_size": 1, "round": args.round,
              "warmup": args.warmup, "iterations": args.iterations,
              "manifest_sha256": manifest_hash, "checkpoint_sha256": export["checkpoint_sha256"],
              "onnx_sha256": export["onnx_sha256"], "threshold": calibration["threshold"],
              "input_sha256": [r["sha256"] for r in selected], "latency": summary(latency),
              "classifier_fps": args.iterations/elapsed, "inference_elapsed_s": elapsed,
              "cpu_time_s": cpu_s, "cpu_percent_one_core_units": 100*cpu_s/elapsed,
              "cpu_percent_total_capacity": 100*cpu_s/elapsed/psutil.cpu_count(),
              "memory": {"baseline_rss_bytes": rss_before, "loaded_rss_bytes": rss_loaded,
                         "warm_rss_bytes": rss_warm, "inference_peak_rss_bytes": inference.peak,
                         "load_peak_rss_bytes": loading.peak,
                         "process_high_water_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024},
              "startup_including_import_ms": startup_ms, "model_file_bytes": model_file.stat().st_size,
              "model_plus_config_bytes": (model_file.stat().st_size+(args.model/"threshold.json").stat().st_size
                  if args.engine == "pytorch" else sum(p.stat().st_size for p in args.model.iterdir() if p.is_file())),
              "runtime_installed_files": package_footprint(packages),
              "hardware_before": start_hardware, "hardware_after_timing": hardware(),
              "latency_definition": "CPU forward/session.run only, three fixed real test crops; preprocessing, camera, network excluded",
              "fps_definition": "Unthrottled classifier loop throughput, not live camera FPS",
              "latencies_ms": latency}
    if args.evaluate:
        logits = []
        for row in rows:
            logits.append(evaluate(read_tensor(args.data, row)).tolist())
        logits = np.asarray(logits)
        labels = np.array([classes.index(r["person"]) for r in rows])
        result["evaluation"] = classification_metrics(logits, labels, classes, calibration["threshold"])
        result["evaluation"].update(independent_known_test=manifest["independent_known_test"],
                                     unknown_identity_disjoint=manifest["unknown_identity_disjoint"])
        result["test_logits"] = logits.tolist()
        result["test_files"] = [r["file"] for r in rows]
    result["hardware_end"] = hardware()
    save_json(args.output, result)
    print(json.dumps({"engine": args.engine, "round": args.round, "latency": result["latency"],
                      "classifier_fps": result["classifier_fps"], "output": str(args.output)}), flush=True)


def suite(args) -> None:
    if (args.output/"comparison.json").exists():
        raise ValueError("Use a new output directory; do not overwrite measured results")
    args.output.mkdir(parents=True, exist_ok=True, mode=0o700)
    results = {"pytorch": [], "onnx": []}
    env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "OPENBLAS_NUM_THREADS": "1",
           "MKL_NUM_THREADS": str(args.threads)}
    for number in range(args.rounds):
        for engine in (("pytorch", "onnx") if number % 2 == 0 else ("onnx", "pytorch")):
            output = args.output/f"{engine}_round{number+1}.json"
            command = [sys.executable, "-m", "training.benchmark_runtime", "worker", "--engine", engine,
                       "--model", str(args.model), "--checkpoint", str(args.checkpoint), "--data", str(args.data),
                       "--threads", str(args.threads), "--warmup", str(args.warmup), "--iterations", str(args.iterations),
                       "--round", str(number+1), "--output", str(output)]
            if number == 0:
                command.append("--evaluate")
            subprocess.run(command, check=True, env=env)
            results[engine].append(json.loads(output.read_text()))
    report = {"device": platform.node(), "architecture": platform.machine(), "rounds": args.rounds,
              "iterations_per_round": args.iterations, "warmup_per_process": args.warmup,
              "isolation": "A fresh sequential subprocess per engine/round; no process loads both runtimes",
              "order": "Alternate PyTorch/ONNX order between rounds", "engines": {}}
    for engine, runs in results.items():
        values = [v for r in runs for v in r["latencies_ms"]]
        cpu_s = sum(r["cpu_time_s"] for r in runs)
        elapsed = sum(r["inference_elapsed_s"] for r in runs)
        report["engines"][engine] = {"latency": summary(values),
            "classifier_fps": len(values)/elapsed, "cpu_percent_one_core_units": 100*cpu_s/elapsed,
            "cpu_percent_total_capacity": 100*cpu_s/elapsed/psutil.cpu_count(),
            "peak_inference_rss_bytes": max(r["memory"]["inference_peak_rss_bytes"] for r in runs),
            "peak_process_rss_bytes": max(r["memory"]["process_high_water_rss_bytes"] for r in runs),
            "warm_rss_mean_bytes": float(np.mean([r["memory"]["warm_rss_bytes"] for r in runs])),
            "model_file_bytes": runs[0]["model_file_bytes"], "model_plus_config_bytes": runs[0]["model_plus_config_bytes"],
            "runtime_installed_files": runs[0]["runtime_installed_files"], "evaluation": runs[0]["evaluation"],
            "hardware_before": [r["hardware_before"] for r in runs],
            "hardware_end": [r["hardware_end"] for r in runs],
            "per_round_mean_ms": [r["latency"]["mean_ms"] for r in runs]}
    a, b = results["pytorch"][0], results["onnx"][0]
    if a["test_files"] != b["test_files"] or a["input_sha256"] != b["input_sha256"]:
        raise ValueError("Compared inputs differ")
    x, y = np.asarray(a["test_logits"]), np.asarray(b["test_logits"])
    np.testing.assert_allclose(x, y, rtol=1e-3, atol=1e-4)
    classes = json.loads((args.model/"classes.json").read_text())
    decisions = lambda values: [reject(p, classes, a["threshold"])[0] for p in softmax(values)]
    if decisions(x) != decisions(y):
        raise ValueError("Runtime rejection decisions differ")
    report["parity"] = {"samples": len(x), "max_absolute_logit_difference": float(np.max(np.abs(x-y))),
                        "argmax_agreement": float(np.mean(x.argmax(1) == y.argmax(1))),
                        "rejection_decisions_match": True, "allclose_rtol": 1e-3, "allclose_atol": 1e-4}
    report["manifest_sha256"] = a["manifest_sha256"]
    report["checkpoint_sha256"] = a["checkpoint_sha256"]
    report["onnx_sha256"] = a["onnx_sha256"]
    save_json(args.output/"comparison.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["suite", "worker"])
    parser.add_argument("--engine", choices=["pytorch", "onnx"])
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if min(args.threads, args.warmup, args.iterations, args.rounds) < 1:
        parser.error("Threads, warmup, iterations and rounds must be positive")
    if args.action == "worker" and args.engine is None:
        parser.error("Worker needs --engine")
    suite(args) if args.action == "suite" else worker(args)


if __name__ == "__main__":
    main()
