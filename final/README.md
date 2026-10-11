# EdgeFace

Nhận diện thành viên bằng MobileNetV3-Small fine-tune, chạy **ONNX trực tiếp trên Raspberry Pi 4**, điều khiển qua Telegram và MQTT. Không gửi frame/video qua MQTT hoặc Telegram; không dùng cloud inference.

MVP đã triển khai: một thành viên `son` và lớp `Unknown`, video thành viên do người dùng cung cấp, dữ liệu Unknown LFW công khai, GPU RTX 2050 để training. Người dùng đã xác nhận Telegram hoạt động. Âm thanh tắt; thu thập camera qua Telegram là chức năng bổ sung chưa kiểm chứng end-to-end.

Model hiện tại được train tiếp với `dataset/videos/son/2.mp4` và dữ liệu cũ, release `d699b7c900ea` (ONNX `5eff47108a6e`). Bundle hiện tại được chia sẻ bằng Git LFS. Dataset/checkpoint/release cũ chỉ giữ cục bộ để rollback; đánh giá và benchmark phiên bản cả team sẽ thực hiện sau khi đủ dữ liệu.

- [Kiến trúc ngắn gọn cho team](docs/ARCHITECTURE.md)
- [Báo cáo chi tiết kiến trúc và phương pháp — tài nguyên cho team, chưa kèm benchmark](report/ARCHITECTURE_AND_METHODS.md)
- [Môi trường và trạng thái kiểm chứng](docs/ENVIRONMENT.md)
- [MQTT Explorer](docs/MQTT_EXPLORER.md)
- [Tailscale native và tự khởi động WSL/Pi](docs/TAILSCALE.md)
- [Quy tắc repository, file riêng và Git LFS](docs/REPOSITORY.md)

## Kiến trúc

```mermaid
flowchart LR
    Telegram[Telegram Cloud] <-->|Long polling| Bot[Bot · laptop Docker]
    Bot <--> Broker[Mosquitto · laptop Docker]
    Explorer[MQTT Explorer · localhost:4000] <-->|Chỉ đọc| Broker
    Broker <-->|MQTT qua Tailscale VPN| Pi[Pi · MQTT client]
    Camera[USB / CSI camera] --> YuNet[YuNet · crop/căn chỉnh]
    YuNet --> ONNX[MobileNetV3-Small · ONNX CPU]
    ONNX --> Pi
```

## Phần cứng và phần mềm

Pi 4, camera USB hoặc CSI với driver phù hợp; laptop có Docker Compose, SSH, Python 3.10–3.12. GPU tùy chọn. ONNX Runtime 1.20.1 được kiểm chứng trên Pi Debian 12 **aarch64**, Python 3.11. Không chạy setup ARM64 trên Pi 32-bit; cần chọn wheel/build tương thích trước, không tự cài lại OS.

## Clone và Git LFS

```bash
git lfs install
git clone https://github.com/sonnguyen112/AIoT.git
cd AIoT/final
git lfs pull
```

Cài Git LFS trên máy trước khi clone. Nếu đã clone, chạy `git lfs install` và `git lfs pull`. Model ONNX phải là binary thật, không phải file pointer; cấu hình demo và dataset riêng không có trong repository.

## Setup laptop

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[bot,edge,training,test]'
cp .env.example .env       # Chỉ làm khi chưa có .env
chmod 600 .env
```

Điền `TELEGRAM_BOT_TOKEN` trong `.env` cục bộ. `TELEGRAM_ALLOWED_USER_IDS` và `TELEGRAM_ALLOWED_CHAT_IDS` nhận ID số, cách nhau bằng dấu phẩy; để trống là chế độ mở theo lựa chọn MVP hiện tại. Không đưa token/password lên Git/chat. Với GPU, cài torch/torchvision theo [hướng dẫn PyTorch chính thức](https://pytorch.org/get-started/locally/) rồi kiểm tra `torch.cuda.is_available()`; training chọn CUDA tự động hoặc `--device cuda`.

```bash
python -m scripts.setup_mqtt
docker compose --profile bot --profile monitor up -d --build
python -m scripts.fetch_detector
```

Script tự tạo mật khẩu riêng cho bot, Pi, tài khoản monitor và UI; lưu ở `.env` quyền 600, broker chỉ nhận file mật khẩu đã hash. UI: **http://localhost:4000**, đăng nhập bằng `MQTT_EXPLORER_USERNAME/PASSWORD`; tự kết nối broker với `edgeface_monitor`. Không xuất `docker compose config` hoặc `docker inspect` toàn bộ ra log vì có thể chứa secrets.

## Kết nối và setup Pi

**Máy hiện tại đã dùng Tailscale native**, WSL `100.73.104.13`, Pi `100.83.132.53`. Pi đặt `MQTT_HOST=100.73.104.13`, broker giữ loopback và Tailscale Serve chuyển TCP 1883 trong tailnet vào broker. MQTT vẫn có password/ACL. Tailscale tự chạy trên cả hai máy; tác vụ Windows tự mở và giữ WSL khi đăng nhập. SSH tunnel cũ đã tắt. Xem [cài đặt, kiểm chứng và tự khởi động](docs/TAILSCALE.md).

Trước tiên kiểm tra OS, kiến trúc, RAM/storage, camera, SSH và Internet. Dùng SSH key; nếu cần password, nhập tương tác tại terminal:

```bash
ssh-copy-id pi@<PI_IP>
PI_HOST=pi@<PI_IP> bash scripts/deploy_pi.sh
python -m scripts.configure_pi --host pi@<PI_IP> --mqtt-host <WSL_TAILSCALE_IP>
tailscale serve --bg --tcp=1883 tcp://127.0.0.1:1883
```

Khi đổi Wi-Fi, địa chỉ Tailscale vẫn giữ nếu node/state hiện tại còn nguyên và hai máy có Internet. Dùng IP Tailscale Pi cho deployment; inference vẫn chạy cục bộ tại Pi. MQTT Explorer vẫn ở http://localhost:4000.

Đường kết nối MQTT hiện tại dùng Tailscale; các script SSH tunnel cũ đã được dọn khỏi repository. SSH vẫn dùng để quản trị và deploy Pi.

Chụp một frame kiểm tra **lưu tại Pi**:

```bash
ssh pi@<PI_IP> 'cd /home/pi/edgeface && .venv/bin/python -m raspberry_pi.camera'
```

Camera `auto` thử Picamera2/libcamera rồi V4L2; camera USB cũng có thể được libcamera hỗ trợ. Chọn `CAMERA_BACKEND=usb`, `CAMERA_INDEX` khi cần. Pi hiện tại dùng CyberTrack H7 và libcamera USB. Không mở camera khi agent idle.

Pi hiện có IP tĩnh `10.220.139.249/24`, gateway `10.220.139.225`, trên Wi-Fi đang dùng. Profile DHCP gốc được giữ. Đổi subnet cần sửa profile hoặc khôi phục DHCP tại terminal Pi; IP này không bảo đảm dùng ở mọi mạng. Xem [môi trường](docs/ENVIRONMENT.md).

## Chuẩn bị dữ liệu từ video

Đặt video: `dataset/videos/<name>/<session>.mp4`. Mỗi video phải thuộc người đồng ý tham gia; tránh nhiều người cùng frame. Tốt nhất **ít nhất 3 phiên độc lập mỗi thành viên** để chia tập theo phiên.

```bash
python -m training.import_video dataset/videos/<name>/<session>.mp4 --name '<name>' --consent
python -m training.fetch_unknown --identities 60 --per-identity 5
python -m training.dataset audit
python -m training.dataset split
```

YuNet phát hiện mặt trước; crop có xoay theo mắt khi landmarks hợp lệ. Loại frame không/multiple faces, mặt nhỏ, mờ, quá tối/sáng, crop trùng lặp. Blur được đo trên ảnh tham chiếu 224×224 để không phụ thuộc độ phân giải video. Metadata lưu source, session, SHA256 và thời gian trong video. Import lần nữa tạo phiên mới; không import lặp cùng video vào nhiều nhóm dữ liệu.

MVP một video được người dùng yêu cầu:

```bash
python -m training.import_video dataset/videos/son/1.mp4 --name son --consent --single-video-temporal --interval 0.12 --duplicate-hamming 1
python -m training.dataset split --allow-single-video-temporal
```

Chia các đoạn 0–50% / 60–75% / 85–100%; bỏ trống phần giữa. Đây là **đánh giá cùng phiên quay**, không thay thế test trên video mới. Unknown được chia theo danh tính, chưa xuất hiện trong train. LFW là dữ liệu web công khai có nguồn/checksum; không tuyên bố đã xác minh consent cho người trong LFW và không phân phối lại ảnh.

`processed/` đã có thì script từ chối ghi đè: lưu bản cũ sang thư mục riêng trước khi chia lại. `audit --delete-corrupt` chỉ xóa ảnh hỏng khi người dùng gọi rõ tùy chọn đó; mặc định chỉ báo cáo và loại khỏi split. Không commit dataset hoặc ảnh diagnostic.

## Training, calibration, đánh giá và ONNX

Để train tiếp một checkpoint với dữ liệu mới có cùng class order/preprocessing, dùng `--init-checkpoint`, replay training data cũ và chọn output mới. Hai video hiện tại dùng `--purged-temporal` khi import và `--allow-multi-video-temporal` khi chia tập; đây vẫn là test trên các phiên đã có dữ liệu train. Với ít nhất ba phiên độc lập, ưu tiên split theo phiên thay cho fallback này. Dataset revision hiện tại là `dataset/revisions/video2/processed`.

```bash
python -m training.train --data dataset/revisions/video2/processed --output runs/<new_run> --init-checkpoint runs/video2/best_model.pth --head-lr 0.0001 --fine-lr 0.00005 --device auto
```

Các lệnh bên dưới dành cho dataset đầy đủ sau khi chia theo phiên độc lập. Dùng output mới cho từng lần train; hiện chưa chạy lại training hoặc benchmark cho cả team. Không warm-start head hai lớp khi thêm thành viên mới.

```bash
TORCH_HOME="$PWD/models/pretrained" python -m training.train --data dataset/processed --output runs/team --device auto --batch-size 16
python -m training.evaluate calibrate --data dataset/processed --checkpoint runs/team/best_model.pth --threshold runs/team/threshold.json
python -m training.evaluate test --data dataset/processed --checkpoint runs/team/best_model.pth --threshold runs/team/threshold.json --output runs/team/evaluation
python -m training.export_onnx --data dataset/processed --checkpoint runs/team/best_model.pth --threshold runs/team/threshold.json --output models/team
```

Head training đóng băng backbone; giai đoạn sau mở các block cuối, LR nhỏ hơn. Có class-weighted loss, augmentation vừa phải, seed, early stopping và checkpoint tốt nhất theo validation loss. Không dùng test để chọn threshold/hyperparameter. Preprocessing training và inference dùng chung RGB, PIL bilinear 224×224, ImageNet mean/std.

File xuất: `face_classifier.onnx`, `classes.json`, `preprocessing.json`, `threshold.json`, `export_validation.json`. Opset 17, batch cố định 1; checker và so sánh logits/decision trên cùng validation input. Unknown được quyết định bởi lớp Unknown và confidence threshold đã calibration.

Baseline là pretrained ImageNet backbone + head được train, backbone đóng băng (`best_head.pth`); model fine-tune là `best_model.pth`. Không dùng trực tiếp các nhãn ImageNet làm classifier danh tính. Để đánh giá baseline, calibration riêng với checkpoint tương ứng; không dùng threshold của model khác.

## Deploy model và service

```bash
python -m scripts.deploy_model --host pi@<PI_IP>
ssh pi@<PI_IP> 'cd /home/pi/edgeface && bash scripts/install_service.sh'
```

Bundle được upload vào release riêng, kiểm tra ONNX trên Pi rồi kích hoạt symlink `models/active`. Dùng `/stop` rồi `/start` để nạp cập nhật model. Service tự chạy sau reboot ở **idle**:

```bash
ssh pi@<PI_IP> 'sudo systemctl status edgeface --no-pager'
ssh pi@<PI_IP> 'sudo journalctl -u edgeface -n 30 --no-pager'
```

Broker/bot/Tailscale trên laptop phải hoạt động. Secrets Pi chỉ gồm credential MQTT thiết bị, không copy Telegram token. `setup_pi.sh` cài dependencies native trong venv, giữ hệ thống hiện có.

## Telegram demo

| Lệnh | Hành vi |
|---|---|
| `/start` | Pi mở camera, phát hiện/crop mặt và ONNX CPU inference |
| `/stop` | Dừng nhận diện, đóng camera |
| `/status` | Trạng thái thực tế, camera, model, CPU, RAM, nhiệt độ, latency |
| `/members` | Thành viên model và thành viên đã có dữ liệu camera |
| `/help` | Hướng dẫn |
| `/collect <name>` | Chức năng bổ sung: xác nhận consent rồi thu crop camera |
| `/collect_unknown <anonymous-id>` | Mỗi participant Unknown có ID ẩn danh riêng để split theo danh tính |

`/start` phản hồi **“Đang nhận diện...”**. Đứng một mình trước camera; khi nhận diện thành viên, bot gửi tên đầy đủ và confidence/latency thật, tự gửi stop và chỉ báo camera đã dừng sau khi Pi xác nhận. **Unknown không gửi tin nhắn Telegram và vẫn tiếp tục nhận diện**; metadata Unknown vẫn có trong MQTT để quan sát/đánh giá. Lệnh tự stop gắn request ID phiên để không dừng nhầm phiên mới. Cooldown mặc định 30 giây mỗi nhãn MQTT trong phiên; `/start` mới reset cooldown để có phản hồi ngay. Nhiều mặt không tự gán nhãn; không có mặt sẽ thông báo sau thời gian cấu hình. Âm thanh tắt.

Tên hiển thị cấu hình tại `configs/display_names.json`: `son` → `Nguyễn Hồ Trường Sơn`. Model/dataset dùng nhãn ổn định; Telegram dùng tên đầy đủ. Đổi mapping không cần train lại. `AUTO_STOP_ON_KNOWN=true` bật hành vi tự dừng.

Collection là giai đoạn sau theo scope người dùng: target 150 mặc định, progress qua MQTT, timeout/stop và quality filtering. Chưa công bố dataset IoT thật hoàn thành. Đồng bộ bằng `PI_HOST=pi@<PI_IP> bash scripts/sync_dataset.sh`.

## Benchmark và kiểm thử

```bash
python -m pytest
python -m scripts.mqtt_smoke
python -m training.benchmark --image <REAL_FACE_CROP> --checkpoint runs/video2/best_model.pth --output reports/generated/benchmark_video2_laptop.json
# Trên Pi, ở idle và camera không bị tiến trình khác chiếm:
.venv/bin/python -m training.benchmark --image <LOCAL_REAL_FACE_CROP> --camera --output reports/generated/benchmark_video2_camera_pi.json
```

Benchmark có warmup, mean/P50/P95, classifier riêng, detection/capture/pipeline riêng, FPS, process RSS, CPU và nhiệt độ. CPU process dùng đơn vị một core: Pi 4 có thể đạt 400%; không nhầm 333% với vượt quá tổng CPU. Pipeline latency loại trừ MQTT/Telegram và không phải latency từ phơi sáng sensor. FPS benchmark chạy không throttle; agent sampling mặc định 0.15 giây nên FPS khi demo thấp hơn.

PyTorch CPU so với ONNX CPU phải cùng thiết bị/input/threads. Pi không cần PyTorch để chạy runtime; chỉ cài tùy chọn nếu muốn benchmark cùng Pi. Không so GPU laptop với CPU Pi rồi kết luận về ONNX.

## Troubleshooting

- **SSH hostname không resolve trên WSL:** dùng IP Pi, kiểm tra cùng Wi-Fi; host key mới được xác minh/pin, không tắt host-key verification.
- **MQTT offline:** kiểm tra `docker compose ps`, `tailscale status`, `tailscale serve status` trên WSL và mật khẩu đúng role. Command không retained, clean session, TTL và request ID SQLite chống thực thi lại sau reconnect/restart. Device LWT báo offline.
- **Bot không phản hồi:** kiểm tra token, hai allowlist nếu bật, health backend, chỉ một bot long-polling chạy cùng token. Exception logs không in token.
- **Camera bận:** `/stop`, kiểm tra có chương trình khác giữ `/dev/video0` không. Không chạy benchmark camera song song agent recognizing.
- **Camera/model error:** `/status` phân biệt lỗi camera/detector/model; Unknown là dự đoán, không phải lỗi vận hành. Stop hoặc lỗi đều đóng camera.
- **Collection ít ảnh:** thay đổi góc/ánh sáng; duplicate và blur filter có thể loại nhiều ảnh. Không bỏ kiểm tra nhiều mặt để tăng count.
- **GPU trong sandbox báo unavailable:** CUDA đã kiểm chứng ngoài sandbox. Chạy training tại terminal có quyền GPU; nếu GPU unavailable, dùng CPU.

## Dữ liệu và giới hạn

Dataset, checkpoint và config private được gitignore; bundle ONNX đang deploy được chia sẻ bằng LFS; dataset/file quyền 700/600. Camera probe/benchmark crop chứa ảnh thật và chỉ lưu local. `python -m scripts.delete_member <name>` là dry-run; thêm `--confirm-name <name>` mới xóa dữ liệu local. Sau đó retrain model để loại thông tin từ trọng số; xóa cả bản đồng bộ trên laptop/Pi và backups theo chính sách nhóm. Không dùng hệ thống làm kiểm soát truy cập có độ tin cậy cao: **chưa có liveness detection**, ảnh in/màn hình có thể đánh lừa classifier.

Model đang chia sẻ là bản demo một thành viên. Train/test được chia theo đoạn thời gian từ cùng nguồn video nên chưa chứng minh khả năng tổng quát ở phiên mới. Khi đủ dữ liệu team, cần chia theo phiên độc lập, train lại và đo chất lượng/hiệu năng trên Pi. Chưa có liveness detection hoặc đánh giá live-camera đầy đủ có ground truth.

Nguồn: [YuNet/OpenCV](https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html), [MobileNetV3-Small](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html), [ONNX Runtime Python ARM64](https://onnxruntime.ai/docs/get-started/with-python.html), [LFW research paper](https://people.cs.umass.edu/~elm/papers/lfw.pdf), [LFW loader/checksum](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py).
