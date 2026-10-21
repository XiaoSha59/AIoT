"""Render architecture diagrams from deployed models and loss from recorded history.

Run with the training extra installed: python -m scripts.render_report_figures.
The zero tensor only traces tensor shapes; no training/evaluation is performed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import onnx
import torch

from training.model import build_model

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report/assets/figures"
BLUE = "#e8f1fb"
GREEN = "#e5f3ed"
ORANGE = "#fff0d9"
INK = "#18324e"
LINE = "#58738b"


def box(ax, x, y, w, h, text, color=BLUE, fontsize=11):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.1",
                              facecolor=color, edgecolor=LINE, linewidth=1.1))
    ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=fontsize, color=INK,
            linespacing=1.4)


def arrow(ax, start, end, text=None):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                linewidth=1.25, color=LINE))
    if text:
        ax.text((start[0]+end[0])/2+.1, (start[1]+end[1])/2, text,
                fontsize=9, color=INK, ha="left", va="center", backgroundcolor="white")


def canvas(title, subtitle, height=11):
    fig, ax = plt.subplots(figsize=(16, height), layout="constrained")
    ax.set(xlim=(0, 16), ylim=(0, height))
    ax.axis("off")
    ax.text(.35, height-.35, title, fontsize=21, fontweight="bold", color=INK, va="top")
    ax.text(.35, height-.92, subtitle, fontsize=11, color=LINE, va="top")
    return fig, ax


def save(fig, name):
    for extension in ("png", "svg"):
        path = OUTPUT / f"{name}.{extension}"
        metadata = {"Date": None} if extension == "svg" else {}
        fig.savefig(path, dpi=240, facecolor="white", metadata=metadata)
        if extension == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines())+"\n")
    plt.close(fig)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yunet():
    path = ROOT / "models/face_detection_yunet_2023mar.onnx"
    model = onnx.shape_inference.infer_shapes(onnx.load(str(path)))
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim]
              for v in [*model.graph.input, *model.graph.value_info, *model.graph.output]}
    assert shapes["input"] == [1, 3, 640, 640]
    for stride, points in [(8, 6400), (16, 1600), (32, 400)]:
        assert shapes[f"bbox_{stride}"] == [1, points, 4]
        assert shapes[f"kps_{stride}"] == [1, points, 10]
    fig, ax = canvas("YuNet — Mô hình phát hiện khuôn mặt",
                     "Tái dựng từ graph ONNX YuNet 2023mar của EdgeFace • pretrained, không fine-tune", 11)
    ax.text(1.9, 9.45, "BACKBONE", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(6.6, 9.45, "FEATURE PYRAMID", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(12.3, 9.45, "DETECTION HEADS", ha="center", fontsize=12, fontweight="bold", color=INK)
    stages = [
        (8.15, "Ảnh BGR → blob NCHW\n1 × 3 × 640 × 640"),
        (6.95, "Stem / model0\nConv 3×3, s=2 + PW/DW\n16 × 320 × 320"),
        (5.55, "MaxPool + model1–2\n16 → 32 → 64 kênh\n64 × 160 × 160  (stride 4)"),
        (4.05, "MaxPool + model3 / C3\n64 × 80 × 80  (stride 8)"),
        (2.55, "MaxPool + model4 / C4\n64 × 40 × 40  (stride 16)"),
        (1.05, "MaxPool + model5 / C5\n64 × 20 × 20  (stride 32)"),
    ]
    for i, (y, text) in enumerate(stages):
        box(ax, .35, y, 3.25, .86, text, fontsize=10)
        if i:
            arrow(ax, (1.975, stages[i-1][0]), (1.975, y+.86))
    box(ax, 4.65, 6.65, 3.8, 2.15,
        "Khối convolution nhẹ\nPointwise Conv 1×1\n→ Depthwise Conv 3×3\n→ ReLU\nNeck: upsample ×2 và cộng đặc trưng", fontsize=10)
    for y, stride, points in [(4.05, 8, 6400), (2.55, 16, 1600), (1.05, 32, 400)]:
        side = 640//stride
        box(ax, 4.65, y, 3.8, .86, f"P{stride.bit_length()-1}: PW/DW convolution\n64 × {side} × {side}", GREEN, 10)
        arrow(ax, (3.6, y+.43), (4.65, y+.43))
        box(ax, 9.5, y, 5.15, .86,
            f"Head stride {stride}  •  {points:,} vị trí\ncls: 1 | objectness: 1 | bbox: 4 | keypoints: 10", GREEN, 10)
        arrow(ax, (8.45, y+.43), (9.5, y+.43))
        ax.plot([14.68, 15.1], [y+.43, y+.43], color=LINE, lw=1.2)
    arrow(ax, (6.55, 1.95), (6.55, 2.5), "↑ ×2 + C4")
    arrow(ax, (6.55, 3.45), (6.55, 4.0), "↑ ×2 + C3")
    box(ax, 9.5, 6.65, 5.15, 2.15,
        "Hậu xử lý OpenCV FaceDetectorYN\nDecode bbox và 5 landmarks\nScore threshold 0,85 • NMS IoU 0,30\nĐầu ra: bbox + landmarks + detector score\nKhông dự đoán danh tính", ORANGE, 11)
    ax.plot([15.1, 15.1], [1.48, 7.72], color=LINE, lw=1.2)
    arrow(ax, (15.1, 7.72), (14.7, 7.72))
    ax.text(.35, .42,
            "Kích thước đặc trưng minh họa theo graph ONNX 640×640; khi chạy, FaceDetectorYN đặt input theo frame (Pi: 640×480).\n"
            "12 output tensors = 4 loại đầu ra × 3 mức stride. Score detector khác confidence danh tính của MobileNetV3.",
            fontsize=10, color=LINE, va="center")
    save(fig, "yunet_architecture")
    return {"file": str(path.relative_to(ROOT)), "sha256": digest(path),
            "input": shapes["input"], "outputs": {v.name: shapes[v.name] for v in model.graph.output},
            "spatial_shapes_are_reference_graph_not_fixed_camera_size": True}


def mobilenet():
    directory = ROOT / "models/active"
    classes = json.loads((directory / "classes.json").read_text())
    model = build_model(len(classes), pretrained=False).eval()
    torch.set_num_threads(2)
    stages = {}
    with torch.inference_mode():
        x = torch.zeros(1, 3, 224, 224)
        for index, feature in enumerate(model.features):
            x = feature(x)
            stages[str(index)] = list(x.shape)
    count = sum(p.numel() for p in model.parameters())
    assert count == 1524006 and model.classifier[3].out_features == 6
    graph = onnx.load(str(directory / "face_classifier.onnx"))
    assert graph.graph.output[0].type.tensor_type.shape.dim[1].dim_value == len(classes)
    fig, ax = canvas("MobileNetV3-Small — Bộ phân loại thành viên EdgeFace",
                     "ImageNet pretrained → fine-tune sáu lớp • 1.524.006 tham số • ONNX FP32 / batch 1", 12.6)
    blocks = [
        ("Input", "Face crop RGB đã normalize", "1 × 3 × 224 × 224"),
        ("features.0", "Stem Conv 3×3, s=2 + BN + h-swish", "16 × 112 × 112"),
        ("features.1", "Bneck k=3, s=2, SE + ReLU", "16 × 56 × 56"),
        ("features.2–3", "2 Bneck k=3, s=2/1, ReLU", "24 × 28 × 28"),
        ("features.4–6", "3 Bneck k=5, s=2/1/1, SE + h-swish", "40 × 14 × 14"),
        ("features.7–8", "2 Bneck k=5, s=1/1, SE + h-swish", "48 × 14 × 14"),
        ("features.9–11", "3 Bneck k=5, s=2/1/1, SE + h-swish", "96 × 7 × 7"),
        ("features.12", "Conv 1×1 + BN + h-swish", "576 × 7 × 7"),
        ("Pooling", "Global average pooling → flatten", "576"),
        ("Classifier", "Linear 576→1024 • h-swish • Dropout 0,2 • Linear 1024→6", "6 logits"),
    ]
    for i, (name, operation, shape) in enumerate(blocks):
        y = 10.55-i*.98
        color = GREEN if name in {"features.12", "Classifier"} else BLUE
        box(ax, .35, y, 8.45, .74, f"{name}  |  {operation}\nĐầu ra: {shape}", color, 10)
        if i:
            arrow(ax, (4.575, y+.97), (4.575, y+.76))
    box(ax, 9.55, 9.1, 5.65, 2.2,
        "Inverted residual / Bneck\n1×1 expansion (khi cần) → depthwise k×k\n→ squeeze-and-excitation (nếu bật)\n→ 1×1 linear projection\nSkip nếu s=1 và số kênh đầu vào = đầu ra", BLUE, 11)
    box(ax, 9.55, 6.7, 5.65, 1.8,
        "SE: điều chỉnh trọng số theo kênh\nDW: mỗi kênh một spatial filter\nPW 1×1: kết hợp thông tin các kênh\nBN: BatchNorm, k: kernel, s: stride", BLUE, 11)
    box(ax, 9.55, 4.25, 5.65, 1.85,
        "Fine-tune trong EdgeFace\nHead stage: chỉ classifier\nFine stage: features.10 / .11 / .12 + classifier\nBatchNorm running statistics được giữ cố định", GREEN, 11)
    box(ax, 9.55, .75, 5.65, 2.8,
        "Hậu xử lý ngoài graph ONNX\nSoftmax → max confidence\nUnknown nếu argmax = Unknown\nhoặc confidence < 0,58332544\nCòn lại: dung / hung / son / thanh / tuan\nDropout tắt ở chế độ inference", ORANGE, 11)
    arrow(ax, (8.85, 1.73), (9.5, 1.73))
    ax.text(.35, .28,
            "Hình mô tả kiến trúc PyTorch dùng để train; graph ONNX inference có thể gộp Conv/BN và loại Dropout. Không INT8/pruning.",
            fontsize=10, color=LINE)
    save(fig, "mobilenetv3_small_architecture")
    return {"classifier_sha256": digest(directory / "face_classifier.onnx"),
            "parameters": count, "feature_output_shapes": stages, "classes": classes,
            "zero_input_used_only_for_shape_inspection": True}


def loss():
    source = ROOT / "report/experiments/team_20261020_son_live/history.json"
    run = json.loads((source.parent / "training_run.json").read_text())
    history = json.loads(source.read_text())
    epochs = [r["epoch"] for r in history]
    best = min(history, key=lambda r: r["val_loss"])
    assert best["epoch"] == run["best_epoch"]
    first_fine = next(r["epoch"] for r in history if r["stage"] == "fine")
    fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
    ax.axvspan(.5, first_fine-.5, color=BLUE, alpha=.75, label="Train head")
    ax.axvspan(first_fine-.5, max(epochs)+.5, color=GREEN, alpha=.55, label="Fine-tune block cuối + head")
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#215b9a", lw=2, label="Training loss")
    ax.plot(epochs, [r["val_loss"] for r in history], "s-", color="#c76420", lw=2, label="Validation loss")
    ax.scatter([best["epoch"]], [best["val_loss"]], s=210, marker="*", color="#a92c34", zorder=5)
    ax.annotate(f"Best checkpoint: epoch {best['epoch']}\nVal loss = {best['val_loss']:.5e}",
                xy=(best["epoch"], best["val_loss"]), xytext=(7.2, 2.2e-7), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": INK}, color=INK)
    ax.axvline(first_fine-.5, color=LINE, ls="--", lw=1)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Weighted Cross-Entropy loss (log scale)",
           xlim=(.5, max(epochs)+.5), ylim=(1.5e-7, 2e-3), xticks=epochs)
    ax.set_title("EdgeFace — Training loss và validation loss\n"
                 "Warm-start checkpoint team • 641 train / 173 validation • RTX 2050", fontsize=15, pad=14)
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.supxlabel("Nguồn: history.json của run team_20261020_son_live; chọn epoch bằng validation, không dùng test.\n"
                  "Training có augmentation/Dropout; validation chạy eval. Loss thấp không chứng minh chất lượng ở phiên quay mới.",
                  fontsize=9, color=LINE)
    save(fig, "training_loss")
    return {"source": str(source.relative_to(ROOT)), "sha256": digest(source),
            "epochs": len(history), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "scale": "log", "interpolation_or_smoothing": False}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "edgeface-report-v1"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {"yunet": yunet(), "mobilenetv3_small": mobilenet(), "training_loss": loss()}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    print(f"Saved PNG/SVG figures and provenance to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
