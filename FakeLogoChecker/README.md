# FakeLogoChecker

## Giới thiệu

**FakeLogoChecker** là một dự án sử dụng mô hình học sâu để phát hiện và phân loại logo giả mạo. Dự án bao gồm hai bước chính:

1. **Phát hiện logo (Localization)**: Sử dụng mô hình YOLOv8m đã được huấn luyện trong 5 epoch để xác định vị trí của các logo trong ảnh.
2. **Phân loại logo (Classification)**: Các logo được cắt ra từ ảnh sau đó sẽ được đưa qua mô hình ResNet50, đã được huấn luyện trong 50 epoch, để phân loại logo là "genuine" (thật) hay "fake" (giả).

Dự án cung cấp một giao diện người dùng đơn giản bằng Gradio để người dùng có thể dễ dàng thử nghiệm và đánh giá kết quả.

## Yêu cầu

Trước khi bắt đầu, bạn cần tải hai mô hình đã huấn luyện và lưu vào thư mục `model` trong dự án:

1. **YOLOv8 model** (sử dụng để phát hiện vị trí của logo trong ảnh):
    Chạy lệnh sau để tải mô hình YOLOv8:
    ```bash
    !wget https://huggingface.co/11pPyLwrFE0VO8A4AUNU_1TBrGQgYodwf -P model/
    ```

2. **ResNet50 model** (sử dụng để phân loại logo là thật hay giả):
    Chạy lệnh sau để tải mô hình ResNet50:
    ```bash
    !wget https://huggingface.co/1ZBBhLoT0hNeS53MP784zXMHPs5FeoDl7 -P model/
    ```

## Cài đặt và chạy ứng dụng

Sau khi tải và lưu các mô hình, bạn có thể chạy ứng dụng bằng lệnh sau:

```bash
python app.py
