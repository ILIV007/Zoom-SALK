# راهنمای اجرای پروژه — Zoom-SALK

این پوشه شامل تمام فایل‌های لازم برای اجرای پروژه است.

## پیش‌نیازها
1. Python 3.10+ (توصیه: 3.12)
2. pip

## مراحل اجرا

### 1. نصب وابستگی‌ها
```bash
pip install -r requirements.txt
```

### 2. بررسی وابستگی‌ها
```bash
python dependency_check.py
```

### 3. دانلود وزن
```bash
gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW -O weights/resnet50-zoomnext.pth
```

### 4. اجرای inference (تک تصویر)
```bash
python infer.py --image <path_to_image> --weight weights/resnet50-zoomnext.pth
```

### 5. اجرای inference با TTA
```bash
python infer_tta.py --image <path_to_image> --weight weights/resnet50-zoomnext.pth
```

### 6. ارزیابی batch با معیارها
```bash
python batch_eval.py
```

## وزن
- فایل: resnet50-zoomnext.pth (114 MB)
- SHA256: e7f958ce379ae709494999d62f489a3bce4cf7c2852f532e7db376abd8119ab9
- دانلود: gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW

## مدل
- RN50_ZoomNeXt (28,458,084 پارامتر)
- Backbone: ResNet-50
