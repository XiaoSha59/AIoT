# Môi trường đã kiểm tra — cập nhật 2026-10-20

Ngày 20/10 đã xác minh: SSH key qua Tailscale, Wi-Fi dùng DHCP (`edgeface-static` autoconnect `no`), model 6 lớp release `5010dc47644c` chạy ONNX Runtime ARM64; MQTT members/status phản hồi đúng. Training GPU và đối chiếu test laptop/Pi được ghi tại [báo cáo team](../report/TEAM_TRAINING_20261020.md).

| Thành phần | Quan sát thực tế |
|---|---|
| Laptop | WSL2 Ubuntu 24.04, x86_64, Python 3.12.14, RAM WSL khoảng 7.6 GiB |
| GPU | NVIDIA RTX 2050, VRAM 4 GiB; CUDA khả dụng trong môi trường thực chạy |
| Training đã chạy | PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130; run team ngày 20/10 mất 196.07 giây trên RTX 2050, 22 epoch thực chạy, best epoch 17 |
| Docker | Engine 29.1.3, Compose v5.6.0; broker/bot/explorer chạy trên laptop |
| Pi | Raspberry Pi 4, Debian 12 Bookworm, kernel 6.12.25+rpt-rpi-v8, aarch64 |
| Python Pi | /usr/bin/python3 3.11.2, venv riêng dùng system-site-packages cho Picamera2 |
| RAM/storage Pi | Khoảng 3.7 GiB RAM, khoảng 14 GiB trống khi kiểm tra ban đầu |
| Camera | USB CyberTrack H7, /dev/video0; 640×480 @30 FPS được driver hỗ trợ |
| Camera thực tế | Picamera2/libcamera truy cập USB camera, chụp được frame 480×640×3, format thực MJPEG/RGB được chuyển sang BGR trước YuNet; không có CSI camera được rpicam CLI liệt kê |
| Runtime | OpenCV 4.11.0, ONNX Runtime 1.20.1 CPU trên laptop và Pi |
| Nhiệt độ | Có telemetry nhiệt độ CPU; benchmark phiên bản cả team thực hiện sau |
| Network | SSH key qua Tailscale 100.83.132.53; Wi-Fi DHCP, IP/gateway LAN thay theo mạng; profile tĩnh cũ tắt autoconnect |
| MQTT | Pi kết nối WSL Tailscale 100.73.104.13:1883 → Serve TCP → broker loopback; password/ACL theo role; SSH tunnel cũ đã disable |
| Tailscale native | Version 1.104.1 trên WSL và Pi; WSL 100.73.104.13, Pi 100.83.132.53; cùng tailnet, ping trực tiếp hai chiều 10–11 ms |
| Tự khởi động | tailscaled enabled cả hai máy, Restart=always/5s; Docker WSL enabled; tác vụ Windows “EdgeFace WSL runtime” đang Running, tự mở/giữ WSL khi đăng nhập Windows |
| Telegram | Long polling, người dùng xác nhận nhận lệnh/kết quả hoạt động |

Phiên bản dependency trong `pyproject.toml` là bộ cài ổn định dự kiến (torch 2.5.1/vision 0.20.1); training hiện tận dụng bộ CUDA có sẵn nêu trên. Lưu versions và hash trong `training_run.json`/`export_validation.json`; không giả định đã train bằng phiên bản pin 2.5.1. PyTorch 2.5.1 ARM64 chỉ được cài thêm để benchmark CPU trên Pi.

## IP tĩnh và khôi phục mạng

**Trạng thái hiện tại:** đã chuyển về DHCP và tắt autoconnect `edgeface-static` ngày 20/10 để Pi dùng được khi hotspot đổi subnet. Thông tin IP tĩnh bên dưới là lịch sử cấu hình, không phải địa chỉ LAN cần dùng hiện nay.

NetworkManager đã tạo profile `edgeface-static`, `ipv4.method=manual`, giữ IP lease đang dùng. SSH mới đã kiểm tra sau áp dụng và hủy timer rollback. Profile DHCP gốc còn nguyên UUID `4b468d2c-b6c6-4fe5-97d1-5671c37aa5bd`.

Nếu gateway/subnet Wi-Fi thay đổi, chạy tại terminal Pi:

```bash
sudo nmcli connection down edgeface-static
sudo nmcli connection up uuid 4b468d2c-b6c6-4fe5-97d1-5671c37aa5bd
```

Nếu muốn giữ DHCP sau reboot: `sudo nmcli connection modify edgeface-static connection.autoconnect no`. Không xóa profile hoặc mật khẩu Wi-Fi gốc. Để tránh router cấp IP static này cho thiết bị khác, reserve/exclude `10.220.139.249` trên router nếu router cho phép; Wi-Fi MAC Pi hiện `e4:5f:01:45:9e:d2`. Địa chỉ/gateway chỉ áp dụng mạng hiện tại.

## Trạng thái acceptance

- Đã kiểm chứng: SSH/camera; MQTT reconnect; start/status/members/stop; chống request trùng; train trọng số thật; ONNX checker/parity; inference local Pi; benchmark thực; Telegram người dùng xác nhận.
- Mạng mới: SSH, MQTT camera/inference/start/stop qua Tailscale và UI telemetry đã kiểm chứng sau khi tắt tunnel. Tự khởi động đã xác minh cấu hình enabled/tác vụ Running; chưa reboot Windows/Pi. Xem [TAILSCALE.md](TAILSCALE.md).
- Model video 2: ONNX `5eff47108a6e`, bundle `d699b7c900ea` đã deploy và kiểm thử camera/auto-stop thật. [Kiến trúc và phương pháp](../report/ARCHITECTURE_AND_METHODS.md).
- Model team hiện tại: bundle `5010dc47644c`, ONNX `1e170d7902b8`, 5 thành viên + Unknown; 181 crop test cùng quyết định trên laptop/Pi, camera mở/inference/đóng và MQTT members/status đã kiểm tra ngày 20/10. Đây chưa phải test các phiên quay độc lập.
- Scope MVP: 5 thành viên + Unknown web, không âm thanh. Đã thu 150 crop Sơn từ camera Pi, sửa RGB/BGR MJPEG, dùng 78 train / 19 val / 26 test, bỏ khoảng đệm 27 ảnh; replay dữ liệu cả nhóm khi fine-tune 107.56 giây trên GPU.
- Chưa xác lập: accuracy live-camera nhiều phiên, người lạ có ground truth, liveness, Bluetooth, dữ liệu training camera IoT đáp ứng đủ đồ án. Không đưa các mục này vào danh sách đã hoàn thành.

Bản sửa màu camera và training bổ sung: [SON_CAMERA_UPDATE_20261020.md](../report/SON_CAMERA_UPDATE_20261020.md).
