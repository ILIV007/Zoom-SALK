# Release Notes — v3.0.3

## خلاصه
نسخه نهایی تحویل دانشگاه. سه اصلاح اصلی نسبت به v3.0.2:

### ۱. اصلاح خطای Colab Notebook
خطای `ModuleNotFoundError: No module named 'numpy.strings'` در Cell 10 (Optional Eval) برطرف شد.
علت: scipy نصب‌شده با numpy 2.x کامپایل شده بود، سپس downgrade به 1.26.4 آن را می‌شکست.
راه‌حل: pin کردن scipy==1.13.1، scikit-image==0.22.0، scikit-learn==1.4.2.

### ۲. تمرکز deliverables روی SALK-Net
تمام mentions نام "ZoomNeXt" از فایل‌های deliverables حذف شد (گزارش، Q&A، اسکریپت، پاورپوینت).
پروژه اکنون به‌طور یکپارچه حول SALK-Net (مقاله مورد مطالعه) قاب‌بندی شده و ارزیابی تجربی
به‌عنوان یک مدل COD مبتنی بر ResNet-50 ارائه می‌شود.

### ۳. اصلاح نگارش فارسی و بازطراحی پاورپوینت
- افزودن `<w:bidi/>` (RTL paragraph-level) به تمام فایل‌های docx — متن اکنون راست‌چین است.
- پاورپوینت از ۱۶ اسلاید به ۱۰ اسلاید فشرده بازطراحی شد با تم مدرن (emerald/amber).
- ساختار گزارش بازسازماندهی شد: ۱۸ بخش با تیترهای رنگی و جداول استایل‌دار.

## فایل‌های تحویل
- University_Report_FA.docx / .pdf
- Presentation_FA.pptx / .pdf (10 اسلاید)
- Presentation_Script_FA.docx
- Defense_QA_FA.docx
- Submission_Checklist.md
