# Model Card — RN50_ZoomNeXt

## Model Details
- **Name:** RN50_ZoomNeXt
- **Architecture:** Unified Collaborative Pyramid Network with triplet zoom input
- **Backbone:** ResNet-50 (ImageNet pretrained, via timm)
- **Parameters:** 28,458,084 (28.458M)
- **Paper:** ZoomNeXt (IEEE TPAMI 2024)
- **Repository:** https://github.com/lartpang/ZoomNeXt

## Intended Use
- **Task:** Camouflaged Object Detection (COD) — image segmentation
- **Input:** RGB image (any resolution, resized to 384×384 base + 0.5×/1.5× auxiliary scales)
- **Output:** Grayscale probability map [0, 1] (binary segmentation mask)

## Training (from paper)
- **Dataset:** COD10K-TR (3,040) + CAMO-TR (1,000) = 4,040 images
- **Optimizer:** Adam, lr=0.0001, step decay
- **Epochs:** 150, batch_size=4 (paper) / 8 (some runs)
- **Input size:** 384×384
- **Loss:** BCE + UAL (Uncertainty Awareness Loss)
- **Note:** This project does NOT train. We use pretrained weights only.

## Evaluation
- **Dataset:** COD10K-Test (2,026 images, 390 evaluated in this project)
- **Metrics:** S-measure, Weighted F-measure, MAE, E-measure
- **Results (390 images, CPU):**
  - Sm: 0.8484 (paper: 0.861)
  - Fw: 0.7597 (paper: 0.768)
  - MAE: 0.0369 (paper: 0.026)
  - Em: 0.8976 (paper: 0.926)

## Limitations
- Evaluated on 390-image subset (not full 2,026) due to resource constraints
- CPU-only evaluation (sandbox limitation)
- Higher MAE than paper due to aquatic-animal bias in subset
- Thin objects (Pipefish) and complex backgrounds (SeaHorse) are failure cases

## TTA Modification
- **Method:** Horizontal flip + averaging (2-pass inference)
- **Cost:** 1.96× inference time, 0 new parameters
- **Result (20 images, representative):** All 4 metrics improved
  - Sm: +0.97%, Fw: +0.56%, MAE: -4.64%, Em: +0.29%
  - 75% of images improved individually

## Weight File
- **File:** resnet50-zoomnext.pth
- **Size:** 114,362,834 bytes
- **SHA256:** e7f958ce379ae709494999d62f489a3bce4cf7c2852f532e7db376abd8119ab9
- **Keys:** 799 (all matched with strict=True)

## Ethical Considerations
- This model is for research and educational purposes
- Camouflaged object detection has applications in wildlife conservation, medical imaging, and security
- No personal data is processed
- The model may produce incorrect predictions on extreme camouflage cases

## Citation
```bibtex
@article{pang2024zoomnext,
  title={ZoomNeXt: A Unified Collaborative Pyramid Network for Camouflaged Object Detection},
  author={Pang, Youwei and Zhao, Xiaoqi and Xiang, Tian-Zhu and Zhang, Lihe and Lu, Huchuan},
  journal={IEEE TPAMI},
  volume={46},
  pages={9205--9220},
  year={2024}
}
```
