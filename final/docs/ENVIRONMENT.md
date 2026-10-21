# Môi trường đã kiểm tra — cập nhật 2026-10-20

Ngày 20/10 đã xác minh: SSH key qua Tailscale, Wi-Fi dùng DHCP (`edgeface-static` autoconnect `no`), model 6 lớp release `5010dc47644c` chạy ONNX Runtime ARM64; MQTT members/status phản hồi đúng. Training GPU và đối chiếu test laptop/Pi được ghi tại [báo cáo team](../report/PROJECT_HANDOFF.md).

| Thành phần | Quan sát thực tế |
|---|---|
| Laptop | WSL2 Ubuntu 24.04, x86_64, Python 3.12.14, RAM WSL khoảng 7.6 GiB |
| GPU | NVIDIA RTX 2050, VRAM 4 GiB; CUDA khả dụng trong môi trường thực chạy |
| Training đã chạy | PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130; run mới nhất mất 107.56 giây trên RTX 2050, 13 epoch thực chạy, best epoch 11 |
| Docker | Engine 29.1.3, Compose v5.6.0; broker/bot/explorer chạy trên laptop |
| Pi | Raspberry Pi 4, Debian 12 Bookworm, kernel 6.12.25+rpt-rpi-v8, aarch64 |
| Python Pi | /usr/bin/python3 3.11.2, venv riêng dùng system-site-packages cho Picamera2 |
| RAM/storage Pi | Khoảng 3.7 GiB RAM, khoảng 14 GiB trống khi kiểm tra ban đầu |
| Camera | USB CyberTrack H7, /dev/video0; 640×480 @30 FPS được driver hỗ trợ |
| Camera thực tế | Picamera2/libcamera truy cập USB camera, chụp được frame 480×640×3, format thực MJPEG/RGB được chuyển sang BGR trước YuNet; không có CSI camera được rpicam CLI liệt kê |
| Runtime | OpenCV 4.11.0, ONNX Runtime 1.20.1 CPU trên laptop và Pi |
| Benchmark team | Cùng Pi, FP32 Torch/ORT: mean 74.080/30.732 ms, classifier 13.497/32.529 FPS, peak RSS 396.855/77.332 MiB; 3×200 inference mỗi engine, process tách riêng |
| Nhiệt độ benchmark | Snapshot runtime 55.5–68.7°C, replay tối đa snapshot 70.1°C; frequency quan sát 1800 MHz, throttling flags 0x0 |
| Network | SSH key qua Tailscale 100.83.132.53; Wi-Fi DHCP, IP/gateway LAN thay theo mạng; profile tĩnh cũ tắt autoconnect |
| MQTT | Pi kết nối WSL Tailscale 100.73.104.13:1883 → Serve TCP → broker loopback; password/ACL theo role; SSH tunnel cũ đã disable |
| Tailscale native | Version 1.104.1 trên WSL và Pi; WSL 100.73.104.13, Pi 100.83.132.53; cùng tailnet, ping trực tiếp hai chiều 10–11 ms |
| Tự khởi động | tailscaled enabled cả hai máy, Restart=always/5s; Docker WSL enabled; tác vụ Windows “EdgeFace WSL runtime” đang Running, tự mở/giữ WSL khi đăng nhập Windows |
| Telegram | Long polling, người dùng xác nhận nhận lệnh/kết quả hoạt động |

Phiên bản dependency trong `pyproject.toml` là bộ cài ổn định dự kiến (torch 2.5.1/vision 0.20.1); training hiện tận dụng bộ CUDA có sẵn nêu trên. Lưu versions và hash trong `training_run.json`/`export_validation.json`; không giả định đã train bằng phiên bản pin 2.5.1. PyTorch 2.5.1 ARM64 chỉ được cài thêm để benchmark CPU trên Pi.

## Mạng hiện tại

Pi sử dụng DHCP và tắt autoconnect profile IP tĩnh cũ để hoạt động khi hotspot đổi subnet. Dùng địa chỉ Tailscale cho SSH/MQTT; mỗi mạng Wi-Fi mới vẫn cần cấu hình thông tin kết nối và Internet. Hướng dẫn vận hành nằm trong [TAILSCALE.md](TAILSCALE.md).

## Trạng thái acceptance

- Đã kiểm chứng: SSH/camera; MQTT reconnect; start/status/members/stop; chống request trùng; train trọng số thật; ONNX checker/parity; inference local Pi; benchmark thực; Telegram người dùng xác nhận.
- Mạng mới: SSH, MQTT camera/inference/start/stop qua Tailscale và UI telemetry đã kiểm chứng sau khi tắt tunnel. Tự khởi động đã xác minh cấu hình enabled/tác vụ Running; chưa reboot Windows/Pi. Xem [TAILSCALE.md](TAILSCALE.md).
- Model team hiện tại: bundle `5010dc47644c`, ONNX `1e170d7902b8`, 5 thành viên + Unknown; 181 crop test cùng quyết định trên laptop/Pi, camera mở/inference/đóng và MQTT members/status đã kiểm tra ngày 20/10. Đây chưa phải test các phiên quay độc lập.
- Scope MVP: 5 thành viên + Unknown web, không âm thanh. Đã thu 150 crop Sơn từ camera Pi, sửa RGB/BGR MJPEG, dùng 78 train / 19 val / 26 test, bỏ khoảng đệm 27 ảnh; replay dữ liệu cả nhóm khi fine-tune 107.56 giây trên GPU.
- Chưa xác lập: accuracy live-camera nhiều phiên, người lạ có ground truth, liveness, Bluetooth, dữ liệu training camera IoT đáp ứng đủ đồ án. Không đưa các mục này vào danh sách đã hoàn thành.

Bản sửa màu camera và training bổ sung: [PROJECT_HANDOFF.md](../report/PROJECT_HANDOFF.md).

Benchmark model team hiện tại: [BENCHMARK_20261020.md](../report/BENCHMARK_20261020.md). Chỉ dùng test có sẵn/replay, không đo camera trực tiếp trong đợt này. Agent đã được bật lại sau đo và MQTT xác nhận online/idle.
