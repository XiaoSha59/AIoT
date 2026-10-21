"""Render Vietnamese report figures from models and recorded experiment JSON.

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
from matplotlib.ticker import FuncFormatter
import numpy as np
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
TORCH_COLOR = "#526d86"
ONNX_COLOR = "#13856e"


def vi(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


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
                     "Tái dựng từ đồ thị ONNX YuNet 2023mar • mô hình huấn luyện sẵn, giữ nguyên trọng số", 11)
    ax.text(1.9, 9.45, "TRÍCH XUẤT ĐẶC TRƯNG", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(6.6, 9.45, "HỢP NHẤT ĐA TỈ LỆ", ha="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(12.3, 9.45, "CÁC NHÁNH DỰ ĐOÁN", ha="center", fontsize=12, fontweight="bold", color=INK)
    stages = [
        (8.15, "Ảnh BGR → tensor NCHW\n1 × 3 × 640 × 640"),
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
        "Khối tích chập nhẹ\nPointwise Conv 1×1\n→ Depthwise Conv 3×3\n→ ReLU\nNeck: tăng kích thước ×2 và cộng đặc trưng", fontsize=10)
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
        "Hậu xử lý OpenCV FaceDetectorYN\nGiải mã khung mặt và 5 điểm mốc\nNgưỡng phát hiện 0,85 • NMS IoU 0,30\nĐầu ra: khung mặt + điểm mốc + điểm phát hiện\nKhông dự đoán danh tính", ORANGE, 11)
    ax.plot([15.1, 15.1], [1.48, 7.72], color=LINE, lw=1.2)
    arrow(ax, (15.1, 7.72), (14.7, 7.72))
    ax.text(.35, .42,
            "Kích thước đặc trưng ứng với đầu vào tham chiếu 640×640; khi chạy trên Pi: khung hình 640×480.\n"
            "12 tensor đầu ra = 4 loại × 3 mức stride. Điểm phát hiện khác độ tin cậy phân loại danh tính.",
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
                     "Khởi tạo từ ImageNet, tinh chỉnh sáu lớp • 1.524.006 tham số • ONNX FP32, batch = 1", 12.6)
    blocks = [
        ("Input", "Vùng mặt RGB đã chuẩn hóa", "1 × 3 × 224 × 224"),
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
        "SE: điều chỉnh trọng số theo kênh\nDW: tích chập không gian riêng từng kênh\nPW 1×1: kết hợp thông tin các kênh\nBN: BatchNorm, k: kernel, s: stride", BLUE, 11)
    box(ax, 9.55, 4.25, 5.65, 1.85,
        "Tinh chỉnh trong EdgeFace\nGiai đoạn đầu: chỉ huấn luyện classifier\nGiai đoạn sau: features.10 / .11 / .12 + classifier\nGiữ cố định thống kê trượt của BatchNorm", GREEN, 11)
    box(ax, 9.55, .75, 5.65, 2.8,
        "Hậu xử lý ngoài đồ thị ONNX\nSoftmax → độ tin cậy lớn nhất\nUnknown nếu argmax = Unknown\nhoặc độ tin cậy < 0,58332544\nCòn lại: dung / hung / son / thanh / tuan\nDropout tắt ở chế độ inference", ORANGE, 11)
    arrow(ax, (8.85, 1.73), (9.5, 1.73))
    ax.text(.35, .28,
            "Kiến trúc huấn luyện PyTorch; đồ thị ONNX có thể gộp Conv/BN và loại Dropout. Chưa lượng tử hóa/cắt tỉa.",
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
    ax.axvspan(.5, first_fine-.5, color=BLUE, alpha=.75, label="Huấn luyện classifier")
    ax.axvspan(first_fine-.5, max(epochs)+.5, color=GREEN, alpha=.55, label="Tinh chỉnh các khối cuối và classifier")
    ax.plot(epochs, [r["train_loss"] for r in history], "o-", color="#215b9a", lw=2, label="Loss huấn luyện")
    ax.plot(epochs, [r["val_loss"] for r in history], "s-", color="#c76420", lw=2, label="Loss xác thực")
    ax.scatter([best["epoch"]], [best["val_loss"]], s=210, marker="*", color="#a92c34", zorder=5)
    ax.annotate(f"Checkpoint tốt nhất: epoch {best['epoch']}\nLoss xác thực = {best['val_loss']:.5e}",
                xy=(best["epoch"], best["val_loss"]), xytext=(7.2, 2.2e-7), fontsize=10,
                arrowprops={"arrowstyle": "->", "color": INK}, color=INK)
    ax.axvline(first_fine-.5, color=LINE, ls="--", lw=1)
    ax.set(yscale="log", xlabel="Epoch", ylabel="Cross-Entropy có trọng số (thang log)",
           xlim=(.5, max(epochs)+.5), ylim=(1.5e-7, 2e-3), xticks=epochs)
    ax.set_title("EdgeFace — Diễn biến hàm mất mát\n"
                 "Tiếp tục từ checkpoint trước • 641 ảnh huấn luyện / 173 ảnh xác thực", fontsize=15, pad=14)
    ax.grid(axis="y", which="both", alpha=.2)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.supxlabel("Nguồn: history.json • run team_20261020_son_live • chọn checkpoint bằng tập xác thực.\n"
                  "Huấn luyện có tăng cường dữ liệu/Dropout; xác thực ở chế độ eval. Chưa kiểm thử các phiên quay độc lập.",
                  fontsize=9, color=LINE)
    save(fig, "training_loss")
    return {"source": str(source.relative_to(ROOT)), "sha256": digest(source),
            "epochs": len(history), "best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
            "scale": "log", "interpolation_or_smoothing": False}


def system_architecture():
    fig, ax = canvas("EdgeFace — Kiến trúc triển khai",
                     "Suy luận tại Raspberry Pi; điều khiển và giám sát qua MQTT", 10.5)
    for x, title, color in [(.3, "LAPTOP / WSL2", BLUE), (9, "RASPBERRY PI 4", GREEN)]:
        ax.add_patch(FancyBboxPatch((x, 1.7), 6.7, 5.75, boxstyle="round,pad=0.08",
                                    facecolor=color, edgecolor=LINE, alpha=.5))
        ax.text(x+.2, 7.1, title, fontsize=12, fontweight="bold", color=INK)
    box(ax, .6, 8.25, 3.5, .85, "Người dùng\nỨng dụng Telegram", fontsize=11)
    box(ax, 5.2, 8.25, 4.1, .85, "Telegram Cloud\nBot API", fontsize=11)
    arrow(ax, (4.15, 8.68), (5.15, 8.68))
    arrow(ax, (5.15, 8.4), (4.15, 8.4))
    box(ax, .7, 5.3, 2.8, 1, "Telegram Bot\nLong polling", fontsize=11)
    box(ax, 4.1, 5.3, 2.4, 1, "Mosquitto\nMQTT broker", fontsize=11)
    arrow(ax, (5.7, 8.2), (2.1, 6.35), "Bot API")
    arrow(ax, (2.6, 6.35), (6.2, 8.2))
    arrow(ax, (3.55, 5.95), (4.05, 5.95))
    arrow(ax, (4.05, 5.55), (3.55, 5.55))
    box(ax, 4.1, 3.7, 2.4, .9, "MQTT Explorer\nGiám sát chỉ đọc", fontsize=10)
    arrow(ax, (5.3, 5.25), (5.3, 4.65))
    box(ax, .7, 2.1, 5.8, .85, "PyTorch + GPU → ONNX và cấu hình\nHuấn luyện, đánh giá và xuất mô hình", fontsize=11)
    box(ax, 9.4, 5.3, 5.8, 1, "Edge agent · MQTT client\nCamera, trạng thái, kết quả và telemetry", GREEN, 11)
    arrow(ax, (6.55, 5.95), (9.35, 5.95), "Lệnh qua MQTT")
    arrow(ax, (9.35, 5.5), (6.55, 5.5))
    ax.text(7.95, 5.12, "ACK, sự kiện, số đo", fontsize=9, ha="center", color=LINE)
    box(ax, 9.4, 3.7, 5.8, .9, "Camera → YuNet → vùng mặt\n→ MobileNetV3-Small · ONNX Runtime CPU", GREEN, 11)
    arrow(ax, (12.3, 5.25), (12.3, 4.65))
    box(ax, 9.4, 2.1, 5.8, .85, "Ảnh kết quả trong RAM\nHTTP có token · chỉ truy cập qua Tailscale", ORANGE, 10)
    arrow(ax, (12.3, 3.65), (12.3, 3))
    ax.plot([9.3, 7.55, 7.55, 3.1], [2.5, 2.5, 4.85, 4.85], color=ONNX_COLOR, lw=1.5, ls="--")
    arrow(ax, (3.1, 4.85), (3.1, 5.25))
    ax.text(7.65, 3.3, "JPEG", color=ONNX_COLOR, fontsize=10)
    ax.plot([6.55, 8.2, 8.2], [2.3, 2.3, 3.35], color=LINE, lw=1.2, ls=":")
    arrow(ax, (8.2, 3.35), (9.35, 3.95))
    ax.text(7, 1.65, "Triển khai model: SSH", fontsize=9, color=LINE, ha="center")
    ax.text(.35, .85, "Tailscale bảo vệ đường kết nối laptop–Pi; Telegram sử dụng Internet.\n"
            "MQTT chỉ truyền lệnh và metadata. JPEG đi qua HTTP riêng khi bật thông báo ảnh.", fontsize=11, color=LINE)
    save(fig, "system_architecture")


def data_pipeline():
    fig, ax = canvas("EdgeFace — Từ dữ liệu đến mô hình triển khai",
                     "Tách vùng mặt trước khi huấn luyện; chọn checkpoint và ngưỡng bằng tập xác thực", 9.5)
    columns = [.35, 5.8, 11.25]
    rows = [
        (7, ["Video thành viên\nẢnh Pi của Sơn: vùng mặt đã lọc",
             "LFW · lớp Unknown\nCác danh tính ngoài nhóm",
             "Quy ước chung\nRGB · 224×224 · ImageNet mean/std"]),
        (4.9, ["Video: lấy mẫu → YuNet → lọc\nChia thời gian và tạo purge gap",
               "Các vùng mặt Unknown\nChia theo danh tính",
               "Gộp các tập đã chia\nKnown: thời gian · Unknown: danh tính"]),
        (2.8, ["Huấn luyện MobileNetV3-Small\n641 train · 173 validation",
               "Chọn checkpoint và ngưỡng\nTập test không tham gia lựa chọn",
               "Đánh giá cuối trên 181 ảnh\nXuất ONNX và kiểm tra tương đương"]),
    ]
    for y, labels in rows:
        for x, label in zip(columns, labels):
            box(ax, x, y, 4.4, 1.05, label, GREEN if y == 2.8 else BLUE, 11)
    arrow(ax, (2.55, 6.95), (2.55, 6))
    arrow(ax, (8, 6.95), (8, 6))
    # Keep known and Unknown branches separate until their respective splits.
    ax.plot([4.8, 5.1, 5.1, 12.6], [5.43, 5.43, 6.35, 6.35], color=LINE, lw=1.2)
    arrow(ax, (12.6, 6.35), (12.6, 6))
    ax.text(10.1, 6.5, "Thành viên: chia theo thời gian", fontsize=9, ha="center", color=LINE)
    arrow(ax, (10.25, 5.43), (11.2, 5.43))
    arrow(ax, (13.45, 6.95), (13.45, 6))
    ax.plot([13.45, 13.45, 2.55], [4.85, 4.35, 4.35], color=LINE, lw=1.2)
    arrow(ax, (2.55, 4.35), (2.55, 3.9))
    arrow(ax, (4.8, 3.33), (5.75, 3.33))
    arrow(ax, (10.25, 3.33), (11.2, 3.33))
    box(ax, 4.3, 1.05, 7.4, .85,
        "Triển khai Raspberry Pi · ONNX Runtime CPU\nBundle: mô hình + lớp + tiền xử lý + ngưỡng", ORANGE, 11)
    ax.plot([13.45, 13.45], [2.75, 1.48], color=LINE, lw=1.2)
    arrow(ax, (13.45, 1.48), (11.75, 1.48))
    ax.text(.35, .35, "Known: các tập vẫn dùng chung bản ghi/phiên; Unknown: tách danh tính. Cần kiểm thử phiên độc lập để đánh giá khả năng khái quát.",
            fontsize=10, color=LINE)
    save(fig, "data_pipeline")


def recognition_workflow():
    fig, ax = canvas("EdgeFace — Luồng nhận diện và phản hồi",
                     "Điều khiển theo request_id; chỉ gửi thông báo khi nhận diện thành viên đã đăng ký", 12)
    names = [("TELEGRAM / LAPTOP", 2.6), ("MQTT / ĐIỀU KHIỂN", 8), ("RASPBERRY PI", 13.4)]
    for title, x in names:
        ax.text(x, 10.3, title, ha="center", fontsize=11, fontweight="bold", color=INK)
    steps = [
        (.35, 8.8, "/start từ người dùng", BLUE),
        (5.75, 8.8, "Lệnh start · request_id\nPi trả ACK: recognizing", BLUE),
        (11.15, 8.8, "Mở camera theo lệnh", GREEN),
        (11.15, 6.9, "Lấy khung hình\nYuNet → crop → ONNX", GREEN),
        (11.15, 5, "Một mặt đã được phân loại\nÁp dụng ngưỡng Unknown", GREEN),
        (5.75, 5, "Nếu là thành viên: result\nMetadata và image_id", BLUE),
        (.35, 5, "Bot nhận kết quả\nGửi stop đúng phiên", BLUE),
        (5.75, 3.1, "Lệnh stop có\nexpected_request_id", BLUE),
        (11.15, 3.1, "Đóng camera\nTrả ACK: idle", GREEN),
        (.35, 1.2, "Gửi tin nhắn kết quả\nvà ảnh chú thích qua Telegram", ORANGE),
        (5.75, 1.2, "Bot tải JPEG qua HTTP\nToken + Tailscale", ORANGE),
        (11.15, 1.2, "Ảnh đã lưu trước khi dừng\nCache tạm trong RAM", ORANGE),
    ]
    for x, y, text_, color in steps:
        box(ax, x, y, 4.5, 1, text_, color, 11)
    for start, end in [
        ((4.9, 9.3), (5.7, 9.3)), ((10.3, 9.3), (11.1, 9.3)),
        ((13.4, 8.75), (13.4, 7.95)), ((13.4, 6.85), (13.4, 6.05)),
        ((11.1, 5.5), (10.3, 5.5)), ((5.7, 5.5), (4.9, 5.5)),
        ((10.3, 3.6), (11.1, 3.6)), ((11.1, 1.7), (10.3, 1.7)),
        ((5.7, 1.7), (4.9, 1.7)),
    ]:
        arrow(ax, start, end)
    ax.plot([2.6, 2.6], [4.95, 3.6], color=LINE, lw=1.2)
    arrow(ax, (2.6, 3.6), (5.7, 3.6))
    arrow(ax, (13.4, 3.05), (13.4, 2.25))
    ax.plot([15.75, 15.95, 15.95], [5.5, 5.5, 7.4], color=LINE, lw=1.2, ls="--")
    arrow(ax, (15.95, 7.4), (15.7, 7.4))
    box(ax, .35, 6.9, 9.9, 1,
        "Unknown: tiếp tục lấy mẫu, không gửi tin nhắn hoặc ảnh.\nKhông có mặt / nhiều mặt: không chạy classifier; sự kiện có cooldown.", ORANGE, 11)
    ax.text(.35, .45,
            "Sơ đồ khối mô tả logic chính; Bot gửi tin nhắn kết quả trước ảnh. Lỗi tải ảnh không làm mất kết quả nhận diện.",
            fontsize=10, color=LINE)
    save(fig, "recognition_workflow")


def benchmark_figures():
    source = ROOT / "report/experiments/benchmark_team_20261020/comparison.json"
    replay_source = source.parent / "pipeline_replay_opencv4.json"
    evaluation_source = ROOT / "report/experiments/team_20261020_son_live/evaluation_metrics.json"
    comparison = json.loads(source.read_text())
    replay = json.loads(replay_source.read_text())
    evaluation = json.loads(evaluation_source.read_text())
    engines = [comparison["engines"][key] for key in ("pytorch", "onnx")]
    mib = 1024**2
    panels = [
        ("Độ trễ classifier trung bình", "ms", [e["latency"]["mean_ms"] for e in engines]),
        ("Thông lượng classifier", "ảnh/giây", [e["classifier_fps"] for e in engines]),
        ("RSS cực đại khi suy luận", "MiB", [e["peak_inference_rss_bytes"]/mib for e in engines]),
        ("Kích thước tệp mô hình", "MiB", [e["model_file_bytes"]/mib for e in engines]),
        ("Tệp cài đặt runtime¹", "MiB", [e["runtime_installed_files"]["logical_bytes"]/mib for e in engines]),
        ("Cross-Entropy trên tập test", "Không trọng số", [e["evaluation"]["test_cross_entropy_unweighted"] for e in engines]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 8.8), layout="constrained")
    fig.suptitle("Đối sánh PyTorch và ONNX Runtime trên Raspberry Pi 4\n"
                 "Cùng trọng số và đầu vào · FP32 · 2 luồng · batch = 1", fontsize=17, color=INK)
    for ax, (title, unit, values) in zip(axes.flat, panels):
        bars = ax.bar(["PyTorch", "ONNX Runtime"], values, color=[TORCH_COLOR, ONNX_COLOR], width=.55)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_ylabel(unit, fontsize=10)
        ax.set_ylim(0, max(values)*1.3)
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
        labels = [f"{v:.5e}".replace(".", ",") if max(values)<.001 else vi(v) for v in values]
        ax.bar_label(bars, labels=labels, padding=5, fontsize=10)
        if max(values)<.001:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        else:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: vi(x, 0)))
        ax.spines[["top", "right"]].set_visible(False)
    fig.supxlabel("Nguồn: comparison.json · 600 lần đo/runtime; đánh giá trên 181 ảnh test.\n"
                  "¹ Chỉ tệp thuộc package; loại OS và các phụ thuộc dùng chung/gián tiếp. Không phải dung lượng SD đã giải phóng.",
                  fontsize=10, color=LINE)
    save(fig, "runtime_comparison")

    classes = evaluation["classes"]
    display = {"Unknown": "Unknown", "dung": "Dung", "hung": "Hùng", "son": "Sơn", "thanh": "Thanh", "tuan": "Tuấn"}
    labels = [display[key] for key in classes]
    matrix = np.asarray(evaluation["confusion_matrix"])
    assert matrix.sum() == 181 and matrix.shape == (6, 6)
    assert matrix.tolist() == engines[1]["evaluation"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(8.7, 8), layout="constrained")
    plot = ax.imshow(matrix, cmap="Blues", vmin=0)
    fig.colorbar(plot, ax=ax, shrink=.78, label="Số ảnh")
    ax.set(xticks=range(6), yticks=range(6), xticklabels=labels, yticklabels=labels,
           xlabel="Nhãn dự đoán", ylabel="Nhãn thực tế")
    ax.set_title("Ma trận nhầm lẫn · 181 ảnh test nội bộ", fontsize=15, pad=20, color=INK)
    ax.tick_params(axis="both", labelsize=11)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=13,
                    color="white" if matrix[i, j] > matrix.max()/2 else INK)
    fig.supxlabel("Thành viên: các tập dùng chung bản ghi/phiên, có tách đoạn thời gian.\n"
                  "Unknown: tách danh tính · Kết quả chưa đại diện cho phiên quay độc lập.",
                  fontsize=10, color=LINE)
    save(fig, "confusion_matrix")

    stages = [
        ("Sao chép frame trong RAM", "acquisition", "#98acbd"),
        ("YuNet", "detection", TORCH_COLOR),
        ("Crop, tiền xử lý và từ chối", "crop_preprocess_rejection", "#d79a2d"),
        ("ONNX classifier", "classifier", ONNX_COLOR),
    ]
    fig, ax = plt.subplots(figsize=(12, 4.5), layout="constrained")
    start = 0.0
    for label, key, color in stages:
        value = replay["timings"][key]["mean_ms"]
        ax.barh([0], [value], left=start, height=.5, color=color, label=f"{label}: {vi(value, 3)} ms")
        if value > 5:
            ax.text(start+value/2, 0, vi(value, 2), ha="center", va="center", color="white", fontsize=12)
        start += value
    ax.set(yticks=[0], yticklabels=["Replay\n640×480"], xlabel="Thời gian trung bình mỗi bước (ms)",
           xlim=(0, start*1.04), ylim=(-.5, .5))
    ax.set_title("Phân rã độ trễ pipeline thị giác trên Raspberry Pi 4\n"
                 "Replay khung hình có sẵn · không gồm camera vật lý và mạng", fontsize=15, color=INK, pad=15)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: vi(x, 0)))
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.3), ncol=2, frameon=False, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "pipeline_stages")
    return {"comparison": {"source": str(source.relative_to(ROOT)), "sha256": digest(source)},
            "replay": {"source": str(replay_source.relative_to(ROOT)), "sha256": digest(replay_source)},
            "confusion_matrix": {"source": str(evaluation_source.relative_to(ROOT)), "sha256": digest(evaluation_source)},
            "reran_training_or_benchmark": False}


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none",
                         "svg.hashsalt": "edgeface-report-v1"})
    OUTPUT.mkdir(parents=True, exist_ok=True)
    provenance = {"yunet": yunet(), "mobilenetv3_small": mobilenet(), "training_loss": loss(),
                  "benchmark_figures": benchmark_figures()}
    system_architecture()
    data_pipeline()
    recognition_workflow()
    provenance["workflow_diagrams"] = {
        "type": "implementation_block_diagrams_not_measurements",
        "sources": {name: digest(ROOT / name) for name in [
            "raspberry_pi/agent.py", "telegram_bot/bot.py", "edgeface/protocol.py",
            "training/train.py", "docker-compose.yml"]}}
    (OUTPUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2)+"\n")
    print(f"Saved PNG/SVG figures and provenance to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
