# Project Understanding Report
## Camouflaged Object Detection using Deep Learning

| Field | Value |
|-------|-------|
| **Phase** | 0 — Project Understanding |
| **Status** | Complete (awaiting uploaded files to begin Phase 1) |
| **Date** | 2026-07-22 (Wednesday), Asia/Tehran |
| **Role** | Senior Computer Vision Research Engineer / Academic Project Assistant |
| **Action taken** | None (no code executed, no files analyzed) |

---

## 1. Project Title

**Camouflaged Object Detection using Deep Learning**

---

## 2. Project Structure (Two Connected Parts)

The project is intentionally split into a theoretical part and a practical part:

- **Part 1 — Academic Research Analysis**
  Analysis of the main paper:
  *"A ternary encoding network fusing scale awareness and large kernel attention for camouflaged object detection"* — referred to as **SALK-Net**.
  Topic: **Camouflaged Object Detection (COD)**.

- **Part 2 — Practical Implementation**
  Based on the open-source repository **ZoomNeXt** (Unified Collaborative Pyramid Network for COD):
  https://github.com/lartpang/ZoomNeXt
  ZoomNeXt is selected because it provides official public code, pretrained weights, and inference support — making it suitable for a university implementation project.

---

## 3. Relationship Between SALK-Net and ZoomNeXt (Critical Clarification)

This distinction must be maintained throughout the entire project and in every deliverable:

> **"SALK-Net and ZoomNeXt solve the same Computer Vision problem (Camouflaged Object Detection) using different architectures. Due to the availability of an official implementation and pretrained weights, ZoomNeXt is selected as the practical implementation framework."**

- ZoomNeXt is **NOT** the same architecture as SALK-Net.
- ZoomNeXt is **NOT** a reproduction of SALK-Net.
- The paper (SALK-Net) provides the **theoretical understanding**; the repository (ZoomNeXt) provides the **executable framework**.

---

## 4. Final Project Goals

The completed project must achieve all of the following:

1. Understand and explain the research paper.
2. Analyze the open-source implementation.
3. Successfully execute the model.
4. Run inference on test images.
5. Make one small, defendable modification.
6. Collect experimental results.
7. Prepare a professional university report.
8. Prepare presentation material.

---

## 5. Core Constraints

This is **NOT** a research reproduction project. The guiding philosophy is:

> **"Successful execution + small improvement + scientific explanation."**

**Forbidden:**
- Training a model from scratch.
- Modifying the whole architecture.
- Redesigning the network.
- Large-scale experiments.
- Spending unnecessary GPU resources.
- Pretending a result exists without running it.
- Inventing experimental numbers.
- Claiming unsupported improvements.

**Required:**
- Preserve the original repository.
- Keep changes minimal and isolated.
- Document every important decision.
- Ask before risky actions.

---

## 6. Execution Environments

Two environments will be used, each with a distinct purpose:

| Environment | Purpose | Hardware | Key Tasks |
|-------------|---------|----------|-----------|
| **1. Google Colab** | Development & testing | GPU if available | Install dependencies, run inference, verify correctness |
| **2. Student Laptop** | University presentation | **CPU only** | Final demo must run without GPU |

**Implication for model choice:** Because the final demo runs on a CPU-only laptop, a **lightweight backbone is preferred** — e.g. **RN50_ZoomNeXt** — rather than the heaviest variants.

---

## 7. Phase Roadmap

| Phase | Name | Output Deliverable | Status |
|-------|------|--------------------|--------|
| 0 | Project Understanding | Project Understanding Report | ✅ In progress |
| 1 | Paper Analysis (SALK-Net) | SALK-Net Analysis Report | ⏳ Awaiting uploaded files |
| 2 | Repository Audit (ZoomNeXt) | ZoomNeXt Repository Audit Report | Pending |
| 3 | Execution | Execution Report | Pending (after approval) |
| 4 | Modification Proposal | Modification Proposal Report | Pending (after approval) |
| 5 | Experiment | Experiment Report | Pending |
| 6 | Documentation | University Report (DOCX/PDF) | Pending |
| 7 | Presentation | Slides, speaker notes, demo scenario | Pending |

### Pipeline of execution (Phase 3 onwards)
```
Input image
   │
   ▼
Preprocessing
   │
   ▼
Model (ZoomNeXt + pretrained weights)
   │
   ▼
Prediction (camouflage mask)
   │
   ▼
Output mask (saved / visualized)
```

---

## 8. Operating Principles (General Rules)

- **Explain before doing.**
- **Document every important decision.**
- **Report errors clearly.**
- **Ask before risky actions.**
- **Preserve the original repository.**
- **Keep changes minimal.**
- Never pretend a result exists without running it.
- Never invent experimental numbers.
- Never claim unsupported improvements.
- Never skip analysis phases.

---

## 9. Current Status & Next Action

- **Current phase:** 0 — Project Understanding.
- **Files analyzed so far:** None.
- **Code executed so far:** None.
- **Next trigger:** The user uploads the **SALK-Net paper** (and any related material such as the ZoomNeXt repository snapshot if provided).
- **Next action upon file receipt:** Begin **Phase 1 (Paper Analysis)** only.
- **Boundary:** I will **NOT** automatically advance to Phase 2 or later. Each phase boundary requires explicit continuation.

---

## 10. Confirmation

I have read and fully understood the project instruction, including:

- The two-part structure (paper analysis + ZoomNeXt implementation).
- The non-equivalence of SALK-Net and ZoomNeXt.
- The constraints (no training, no architecture redesign, CPU-only final demo).
- The phased workflow and the requirement to wait for approval / files before each step.
- The obligation to document decisions and never fabricate results.

I am ready to begin **Phase 1 — Paper Analysis** as soon as the input files (the SALK-Net paper) are uploaded.

---

*End of Project Understanding Report — Phase 0.*
