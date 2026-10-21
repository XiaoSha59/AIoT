# EdgeFace — Benchmark định lượng trên Raspberry Pi 4

Ngày thực hiện: **20/10/2026**. Model hiện hành **bundle `5010dc47644c`**, ONNX `1e170d7902b8`, năm thành viên và Unknown. Báo cáo này bổ sung đối sánh **cùng Pi, cùng trọng số và cùng input** cho rubric Latency/FPS, kích thước trước/sau tối ưu, Accuracy/Loss/F1 và RAM/Flash.

**Kết quả chính:** ONNX nhanh hơn **2.41×**, RSS peak trong classifier thấp hơn **80.51%**, file model nhỏ hơn **1.82%**, không thay đổi quyết định trên 181 ảnh test. Đây là FP32 → FP32; chưa quantize INT8 hoặc pruning. Không dùng latency GPU laptop làm đối chứng CPU Pi.

## 1. Audit theo rubric

| Nội dung | Trước đợt đo này | Đã bổ sung |
|---|---|---|
| Latency mean/P50/P95 | Có một lượt xác minh ONNX cho model team; benchmark đối sánh cũ dùng model hai lớp | 600 inference mỗi runtime, 3 tiến trình/lượt độc lập trên cùng Pi |
| FPS | Thiếu bảng classifier FPS cho model team hiện tại | Classifier throughput đo bằng thời gian loop, tách khỏi replay pipeline FPS |
| Kích thước trước/sau | Có ONNX size riêng lẻ | PyTorch checkpoint ↔ ONNX, bundle và installed-runtime footprint |
| Accuracy/F1 | Có test sau fine-tune/export | Chạy lại 181 test input trên cả PyTorch Pi và ONNX Pi; thêm test CE loss |
| RAM | Có RSS của quá trình xác minh; script cũ có lúc cùng nạp hai runtime | Tách process cho từng engine, warm RSS, inference peak và kernel high-water |
| Pipeline | Có smoke camera nhưng không phải bảng benchmark cùng release | Replay frame thuộc đoạn test đã có: YuNet, preprocess, ONNX, FPS, CPU/RAM, chi phí annotation |

Đã có đủ **nhóm số đo mà mục rubric này yêu cầu**. Chất lượng dữ liệu test và khả năng khái quát vẫn có giới hạn ở mục 8; không suy ra chắc chắn đạt trọn điểm từ việc có bảng số liệu.

## 2. Model và bước tối ưu thực tế

- MobileNetV3-Small, **1.524.006 parameters**, classifier 6 output; cùng checkpoint fine-tune, không train hoặc calibration lại trong benchmark.
- Trước: PyTorch eager CPU, `eval` và `inference_mode`, trọng số FP32.
- Sau: export ONNX opset 17, IR 8, fixed input `1×3×224×224`, constant folding; ONNX Runtime `CPUExecutionProvider`, graph optimization `ORT_ENABLE_ALL`.
- PyTorch source có **34 BatchNorm2d modules**; graph ONNX đã xuất có **0 BatchNormalization nodes**. Conv/BatchNorm ở inference đã được gộp trong quá trình export. Graph có 108 initializer FLOAT và không có QuantizeLinear/DequantizeLinear; không gọi đây là INT8.
- Ngưỡng từ validation giữ nguyên **0,5833254400**, preprocessing RGB/PIL bilinear/ImageNet giống nhau.

Checkpoint SHA256: `af8fcef57f90d4315347b0c0a72c06cc0d25d636ed0c1675e5a305b2015d8175`.

ONNX SHA256: `1e170d7902b8836209030c3f9e166dfa1c31bc42a570e37d82294e80d6d15f81`.

Manifest SHA256: `e49dc56d80cda94e6f5132d613b972d0db4f8dfd8ea9b987a1dc777a2b2f88dd`.

## 3. Môi trường và phương pháp đo

Raspberry Pi 4, aarch64, Debian 12/kernel `6.12.25+rpt-rpi-v8`, RAM khoảng 3,7 GiB; CPU frequency quan sát **1800 MHz**, governor `ondemand`. Native Python 3.11.2. PyTorch **2.5.1**, torchvision **0.20.1**, ONNX Runtime **1.20.1**, NumPy **1.26.4**. Phiên bản PyTorch trên Pi khác bộ CUDA 2.14.0 dùng để training trên laptop; provenance và parity được kiểm tra thực tế.

Tạm dừng `edgeface.service` trong lúc đo để tránh camera/background inference cạnh tranh CPU. Mỗi engine/lượt chạy trong **process mới**, tuần tự, không nạp cả PyTorch và ORT vào một tiến trình rồi so RAM. Sau đo đã bật lại service và MQTT xác nhận online/idle, camera đóng, không có lỗi.

| Thiết lập | Giá trị |
|---|---|
| Runtime threads | Intra-op 2, inter-op 1, CPU |
| Batch/input | 1, float32, 3×224×224 |
| Số lượt | 3 mỗi engine |
| Warmup | 30 mỗi process/lượt |
| Đo chính | 200 inference/lượt, tổng 600/engine |
| Thứ tự | Torch → ONNX; ONNX → Torch; Torch → ONNX |
| Input latency | Ba crop thật từ tập test, deterministic, cùng SHA256 trên cả hai engine; lặp theo chu kỳ |
| Quality input | Toàn bộ 181 ảnh test; xác minh SHA256 và manifest |
| RSS | Sampler mỗi 20 ms trong inference; thêm Linux `ru_maxrss` cho peak qua load/warmup/inference |
| Lưu trữ | Bytes file thật, không dùng kích thước Git LFS pointer |

Không nạp toàn bộ 181 tensor cùng lúc để tránh cộng hàng trăm MiB dataset cache vào RAM runtime. Input latency chỉ chiếm khoảng 1,72 MiB, giống nhau ở hai process. Quality evaluation đọc từng ảnh; chạy sau phần timing. Latency chỉ bọc `forward`/`session.run`, loại camera, đọc file, resize/normalize và network. FPS classifier là số inference chia thời gian thực của loop, không phải FPS camera hay `1 / P95`.

Nhiệt độ quan sát từ **55,5–68,7°C** trong các lượt runtime; các lần đọc `vcgencmd get_throttled` đều **0x0**, frequency quan sát 1800 MHz. Các giá trị là snapshot đầu/cuối, không khẳng định không có dao động giữa hai thời điểm. Báo cáo giữ per-round timings để thể hiện biến thiên nhiệt/thứ tự.

## 4. Đối sánh runtime, RAM và dung lượng

Đơn vị **MiB = 1.048.576 bytes**; phân biệt dung lượng file trên SD với RSS trong RAM.

| Metric | Trước: PyTorch CPU FP32 | Sau: ONNX Runtime CPU FP32 |
|---|---:|---:|
| Classifier latency mean (ms) | 74.080 | 30.732 |
| Classifier latency P50 (ms) | 74.331 | 30.720 |
| Classifier latency P95 (ms) | 75.564 | 31.371 |
| Classifier FPS | 13.497 | 32.529 |
| File model (bytes) | 6,218,949 | 6,105,639 |
| File model (MiB) | 5.931 | 5.823 |
| Model + metadata cấu hình (MiB) | 5.960 | 5.853 |
| Warm RSS trung bình (MiB) | 395.095 | 76.242 |
| RSS peak trong inference (MiB) | 396.855 | 77.332 |
| Kernel high-water RSS qua load/warmup/inference (MiB) | 397.039 | 77.633 |
| File cài runtime — logical (MiB) | 428.041 | 34.894 |
| File cài runtime — allocated trên filesystem (MiB) | 460.828 | 36.172 |
| CPU process, đơn vị một core (%) | 192.35 | 201.84 |
| CPU process / tổng 4 core (%) | 48.09 | 50.46 |

**Định nghĩa RAM:** số liệu gồm Python, thư viện/runtime, model, tensor và buffer của chính tiến trình classifier; không phải chỉ kích thước weights. `peak_inference` không phải tổng RAM toàn Pi. `process high-water` gồm load/warmup/inference, được chụp trước quality evaluation. Máy vẫn còn khoảng 3 GiB RAM khả dụng; không nhầm available RAM với RAM model sử dụng.

**Định nghĩa Flash/SD:** checkpoint PyTorch đã chứa class/preprocess metadata; bundle trước tính checkpoint + threshold JSON. Bundle ONNX tính model + classes + preprocessing + threshold + export validation. Model `.pth` và `.onnx` có container/layout khác nhau; **mức giảm 1,82% không phải nén 4× hay bằng chứng quantization**.

Runtime footprint đếm các file do `torch` + `torchvision` sở hữu so với `onnxruntime`, có logical và allocated bytes. **Không bao gồm toàn bộ OS hoặc mọi dependency dùng chung/transitive** như NumPy, OpenCV, Pillow, Python. Cả hai runtime hiện vẫn cài trên Pi để benchmark; không gọi chênh lệch bảng là lượng SD đã thực sự giải phóng. Không dùng độ trống cả SD thay cho footprint mô hình.

CPU dùng đơn vị một core có thể lớn hơn 100%; Pi bốn core tương ứng tối đa khoảng 400%. ONNX dùng gần hai core trong thời gian ngắn hơn; không suy từ phần trăm CPU này thành công suất điện hay thời lượng pin.

![Runtime comparison](assets/benchmark_team_20261020/runtime_comparison.png)

### Biến thiên từng lượt

| Lượt | PyTorch mean (ms) | ONNX mean (ms) |
|---|---:|---:|
| 1 | 72.819 | 30.170 |
| 2 | 74.605 | 30.769 |
| 3 | 74.818 | 31.256 |

## 5. Chất lượng trước/sau export và chạy trên Pi

| Metric trên cùng 181 ảnh test | PyTorch FP32 trên Pi | ONNX FP32 trên Pi |
|---|---:|---:|
| Accuracy | 181/181 = 100% | 181/181 = 100% |
| Macro F1 | 1,000 | 1,000 |
| Test CrossEntropy, unweighted | 9.730094219e-05 | 9.730307778e-05 |
| Known-member accuracy | 160/160 | 160/160 |
| Unknown rejection | 21/21 | 21/21 |
| Unknown FAR | 0/21 | 0/21 |

Test gồm Unknown 21, Dung 23, Hùng 30, Sơn 64, Thanh 24, Tuấn 19. Precision/recall/F1 mỗi lớp đều 1 trên mẫu này. **Không có bằng chứng tăng accuracy từ export**, vì mục đích ở đây là giữ nguyên chất lượng trong khi đổi runtime. CrossEntropy dùng logits trước threshold, không class weighting; không so trực tiếp với weighted validation loss của training như cùng một phép đo.

PyTorch CPU Pi ↔ ONNX CPU Pi: argmax agreement **100%**, quyết định sau threshold giống toàn bộ, max absolute logit difference **8.153915405e-05**, đạt allclose rtol 1e-3/atol 1e-4. Test không được dùng để chọn ngưỡng hoặc train lại.

Baseline backbone ImageNet đóng băng + head đã train đã được đánh giá trong [số liệu baseline gốc](experiments/baseline_team_20261020.json): baseline và fine-tune đều 155/155 ở revision cũ. Đó là so sánh phương pháp training, khác với PyTorch ↔ ONNX cùng weights trong bảng này. Không trộn test 155 ảnh cũ với 181 ảnh của revision mới rồi kết luận fine-tune tăng accuracy.

## 6. Replay pipeline từ dữ liệu đã có

Theo yêu cầu người dùng, **không mở camera hoặc yêu cầu chụp thêm ảnh cho benchmark này**. Dùng frame có sẵn ở video Sơn 1, vị trí yêu cầu 16 s — nằm trong đoạn test 85–100% của video 17,47 s — resize giữ tỷ lệ và letterbox tới **640×480**. Không tạo ảnh huấn luyện giả; không dùng 200 lần lặp cùng frame để báo accuracy 200 mẫu độc lập.

Một process mới chạy 20 warmup + 200 frame replay. OpenCV **4 threads**, đúng mặc định hiện hành trên Pi; ORT 2 threads. Tất cả frame có đúng một mặt nên detector và classifier đều thực thi. Kết quả sau đây là throughput tính toán trên replay, **không phải camera-to-result live benchmark**.

| Bước replay | Mean (ms) | P50 (ms) | P95 (ms) |
|---|---:|---:|---:|
| Copy frame RAM, không phải camera capture | 0.425 | 0.424 | 0.488 |
| YuNet detection | 73.847 | 73.351 | 78.461 |
| Crop/alignment + RGB/normalize + rejection | 11.451 | 11.433 | 11.645 |
| ONNX classifier | 31.516 | 31.426 | 31.726 |
| Replay frame → recognition result | 117.239 | 116.641 | 122.491 |

- Throughput replay toàn pipeline: **8.528 FPS**.
- Peak RSS trong pipeline: **121.348 MiB**; kernel high-water cả process tới sau annotation: **123.605 MiB**.
- CPU: **315.11%** theo một core = **78.78%** tổng bốn core.
- Nhiệt độ snapshot: **59.9 → 70.1°C**, throttling flags 0x0.
- Vẽ bbox/tên/confidence và encode JPEG, đo riêng 20 lần trên cùng frame: mean **15.469 ms**, P95 **15.937 ms**, JPEG **67.361 bytes**. Gồm lần render đầu tiên, không warmup annotation riêng; loại phần này khỏi FPS vision ở trên. Không đo HTTP/Tailscale/Telegram upload trong số latency classifier.

![Pipeline stages](assets/benchmark_team_20261020/pipeline_stages.png)

Đo chẩn đoán thêm OpenCV 1 thread trên cùng frame: **5.679 FPS**, end-to-end replay mean **176.068 ms**, CPU khoảng **129.3%** một core. Điều này cho thấy đổi thread count có trade-off tài nguyên/tốc độ; giữ mặc định 4 cho ứng dụng hiện tại, không âm thầm dùng kết quả một cấu hình khác để mô tả deployment.

Agent vận hành còn có `FRAME_INTERVAL_S=0.15`, cooldown và auto-stop sau known. Vì vậy **không tuyên bố app thực tế chạy 32,53 FPS hoặc 8,53 FPS liên tục**. 32,53 là classifier-only; 8,53 là replay pipeline bỏ sampling/auto-stop và không có camera acquisition.

## 7. Artifacts, tái chạy và kiểm thử

[Numeric artifacts](experiments/benchmark_team_20261020/): `comparison.json`, sáu file per-engine/per-round có raw latency arrays, hai replay results, `model_structure.json`, `replay_source.json`, `provenance.json`, trạng thái sau khôi phục service. Không chứa ảnh mặt, checkpoint, token hoặc full prediction examples.

Scripts:

- [`training/benchmark_runtime.py`](../training/benchmark_runtime.py): suite so sánh runtime có process isolation, memory/CPU/Flash và test quality/parity.
- [`training/benchmark_pipeline.py`](../training/benchmark_pipeline.py): replay từ full frame hoặc camera khi người dùng chủ động chọn; phiên này chỉ chạy replay.

Trên Pi, dùng checkpoint/input thật đã đồng bộ riêng:

```bash
cd /home/pi/edgeface
.venv/bin/python -m training.benchmark_runtime suite \
  --model models/releases/5010dc47644c \
  --checkpoint reports/generated/benchmark_team_20261020/best_model.pth \
  --data reports/generated/team_20261020_son_live_test \
  --output reports/generated/<new_runtime_run> \
  --threads 2 --warmup 30 --iterations 200 --rounds 3
.venv/bin/python -m training.benchmark_pipeline \
  --mode replay --frame reports/generated/benchmark_team_20261020/replay_frame.jpg \
  --model models/releases/5010dc47644c --threads 2 --opencv-threads 4 \
  --warmup 20 --iterations 200 --output reports/generated/<new_pipeline_run>.json
```

Chọn output mới, không ghi đè các lần đo trước. Tạm dừng agent nếu nó đang dùng CPU/camera và khôi phục sau phép đo. Bộ kiểm thử hiện **60 passed**, gồm test percentile, threshold-aware F1/accuracy/loss và từ chối benchmark khi calibration/manifest sai provenance. Fixture unit test không tham gia huấn luyện hoặc báo cáo accuracy.

## 8. Giới hạn cần trình bày khi bảo vệ

1. **Test thành viên cùng phiên quay:** `independent_known_test=false`. Có purge gaps, chống duplicate và Unknown tách danh tính, nhưng vẫn thiếu test thành viên ở phiên/điều kiện mới. 100%/F1=1 không phải cam kết live accuracy.
2. **Unknown ít và khác domain:** 21 ảnh web Unknown, không đại diện mọi người lạ trước camera Pi; FAR 0/21 là ước lượng trên tập nhỏ.
3. **Runtime timing dùng ba crop test cố định, không đo độ trễ phân phối trên mọi bối cảnh camera.** Mạng CNN fixed shape không có nhánh phân loại thay đổi số lớp; toàn bộ 181 ảnh được dùng riêng cho quality/parity.
4. **Không có camera live FPS trong đợt này**, theo yêu cầu dùng dữ liệu có sẵn. Copy frame RAM không thay thế capture sensor, driver buffering hoặc end-to-end tới Telegram.
5. **Benchmark ngắn hạn:** ba lượt runtime và khoảng 23,45 s replay không thay thế bài thermal soak dài; snapshot không chứng minh không có transient throttling.
6. Chưa INT8/pruning, không đo điện năng hoặc khả năng chống giả mạo. Hệ thống chưa có liveness detection.

Các giới hạn này không làm mất giá trị của đối sánh cùng Pi/cùng weights, nhưng phải đi cùng bảng số liệu trong báo cáo và slide về sau.
