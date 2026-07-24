# راهنمای اجرا در Google Colab

## 1. آپلود فایل‌ها
- فایل‌های Run_Project را در Google Drive آپلود کنید
- یا مستقیماً در Colab آپلود کنید

## 2. نصب وابستگی‌ها
```python
!pip install opencv-python-headless timm==0.9.12 einops gdown
!pip install "numpy<2"
```

## 3. دانلود وزن
```python
!gdown 1eN2EtzCKAlqLrN1NlkDi_v17dVnymLoW -O weights/resnet50-zoomnext.pth
```

## 4. اجرای inference
```python
!python infer.py --image test.jpg --weight weights/resnet50-zoomnext.pth
```

## 5. اجرای TTA
```python
!python infer_tta.py --image test.jpg --weight weights/resnet50-zoomnext.pth
```

## 6. اجرای بنچمارک کامل
```python
!python batch_eval.py
```

## نکته
- Runtime → Change runtime type → GPU را انتخاب کنید
- torch پیش‌نصب‌شده Colab را دوباره نصب نکنید
