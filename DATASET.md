# Dataset — COD10K-Test

## Overview
COD10K is the standard benchmark dataset for Camouflaged Object Detection, introduced by Fan et al. (CVPR 2020).

## Statistics
| Split | Images | Usage |
|-------|--------|-------|
| COD10K-Train | 3,040 | Training (used by paper, not by this project) |
| CAMO-Train | 1,000 | Training (used by paper, not by this project) |
| COD10K-Test | 2,026 | Testing (390 evaluated in this project) |
| CHAMELEON | 76 | Testing (not used in this project) |
| CAMO-Test | 250 | Testing (not used in this project) |
| NC4K | 4,121 | Testing (not used in this project) |

## Categories (10 main, 65 sub)
1. Aquatic (BatFish, Octopus, Pipefish, LeafySeaDragon, etc.)
2. Terrestrial (Cat, Deer, Fox, etc.)
3. Flying (Owl, Moth, etc.)
4. ... (see COD10K paper for full list)

## Data Format
- **Images:** .jpg, RGB, variable resolution (500×281 to 1365×1024)
- **GT Masks:** .png, binary (0=background, 255=camouflaged object)
- **Image naming:** COD10K-CAM-{class}-{category}-{subcategory}-{animal}-{number}.jpg

## Integrity Verification
- Total images: 2,026
- Total GT masks: 2,026
- Matched pairs: 2,026 (100%)
- Corrupt files: 0
- Zero-size files: 0
- GT format: 100% binary (0/255)
- Validated with: `python run/dependency_check.py` and `validate_dataset.py`

## Download
```bash
# COD Test Dataset (all 4 test sets, 1.3 GB)
gdown 1V0iSEdYJrT0Y_DHZfVGMg6TySFRNTy4o -O cod_test.zip
unzip cod_test.zip
```

## Preprocessing
- Multi-scale resize: 0.5× (192×192), 1.0× (384×384), 1.5× (576×576)
- Normalization: /255 → ImageNet mean (0.485, 0.456, 0.406) / std (0.229, 0.224, 0.225)
- No test-time augmentation (except TTA modification, which is separate)

## Subset Evaluated
- First 390 images alphabetically
- Predominantly aquatic animals (BatFish, Octopus, Pipefish, LeafySeaDragon, SeaHorse, Crab, etc.)
- This subset is representative but biased toward harder cases (aquatic camouflage)
- MAE is higher than paper due to this bias

## Citation
```bibtex
@inproceedings{fan2020camouflaged,
  title={Camouflaged object detection},
  author={Fan, Deng-Ping and Ji, Ge-Peng and Sun, Guolei and Cheng, Ming-Ming and Shen, Jianbing and Shao, Ling},
  booktitle={CVPR},
  year={2020}
}
```
