# Table 7 — Ablation: Decoder Structure (COD10K)

**Source:** SALK-Net paper, Page 15 (Table 7)

P = PDC (Partial Decoder Component, from SINet)
N = NCD (Neighbor Connection Decoder, from SINet-V2)
C = CID (Cross-Interleaved Decoder, proposed)
CM = CIEM (Channel Interaction and Enhancement Module, proposed)

| No. | P | N | C | CM | Sm ↑ | Fω ↑ | M ↓ | Em ↑ |
|-----|---|---|---|----|------|------|-----|------|
| ① | ✓ | | | | 0.840 | 0.730 | 0.030 | 0.898 |
| ② | | ✓ | | | 0.842 | 0.735 | 0.029 | 0.901 |
| ③ | | | ✓ | | 0.845 | 0.744 | 0.029 | 0.905 |
| ④ | | | ✓ | ✓ | **0.852** | **0.746** | **0.027** | **0.908** |

**Key observations:**
- The proposed CID decoder (③) outperforms both PDC (①) and NCD (②) on all metrics.
- Adding CIEM on top of CID (④) gives a further improvement, confirming the effectiveness of channel-interaction enhancement.
- This justifies the **Cross-Interleaved Decoder + CIEM** combination used in the final architecture.
