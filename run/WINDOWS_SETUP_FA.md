# راهنمای نصب و اجرا در ویندوز

## 1. نصب Python
- Python 3.12 را از python.org دانلود و نصب کنید
- تیک "Add Python to PATH" را بزنید

## 2. ساخت Virtual Environment
```cmd
python -m venv venv
venv\Scripts\activate
```

## 3. نصب PyTorch (CPU)
```cmd
pip install torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
```

## 4. نصب وابستگی‌ها
```cmd
pip install -r requirements.txt
```

## 5. دانلود وزن
```cmd
pip install gdown
gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW -O weights\resnet50-zoomnext.pth
```

## 6. اجرای inference
```cmd
python infer.py --image test.jpg --weight weights\resnet50-zoomnext.pth
```

## 7. اجرای TTA
```cmd
python infer_tta.py --image test.jpg --weight weights\resnet50-zoomnext.pth
```

## خطاهای رایج
- **ModuleNotFoundError: No module named 'torch'**: venv را فعال کنید
- **_ARRAY_API not found**: `pip install "numpy<2"` را اجرا کنید
- **CUDA not available**: در ویندوز CPU-only این طبیعی است
