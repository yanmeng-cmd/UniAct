# UniAct

FPGA activation functions with a golden model, LUT coefficient search, and five model evaluations.

## Directory Structure

```text
.
├── Makefile               # Setup, downloads and evaluation
├── README.md              # Usage
└── src/
    ├── model/
    │   ├── golden.py          # Bit-accurate golden model
    │   ├── uniact.py          # PyTorch activation implementation
    │   ├── lfst.py            # LUT coefficient search
    │   ├── coefficients.json  # Shared LUT coefficients and INIT values
    │   ├── bert.py            # BERT-large / GLUE
    │   ├── deit.py            # DeiT-Tiny / ImageNet
    │   ├── efficientnet.py    # EfficientNet-B0 / ImageNet
    │   ├── vision.py          # Shared ImageNet evaluator
    │   ├── mlgcn.py           # ML-GCN / COCO
    │   ├── tinyllama.py       # TinyLlama / WikiText-2
    │   ├── download.py        # Model and dataset downloads
    │   ├── run.py             # Evaluation entry point
    │   └── requirements.txt  # Python dependencies
    └── rtl/
        ├── wrapper1.v        # Single-core top
        ├── wrapper16.v       # 16-core top
        ├── array16.v
        ├── core.v
        ├── lut.v
        ├── exp_round.v
        ├── reciprocal.v
        ├── dsp.v
        ├── tb_wrapper1.sv
        └── tb_wrapper16.sv
```

## Software

Linux, Python 3.10–3.12, Make and Git.

Install the environment, then download the models and datasets:

```bash
make setup
make download
```

### Model Evaluation

Evaluate all five models:

```bash
make all
```

Or evaluate a single model:

```bash
make bert-large
make deit-tiny
make efficientnet-b0
make mlgcn-coco
make tinyllama
```

### LUT Coefficients

Generate shared LUT coefficients and INIT values from the fixed initial tables:

```bash
make lfst
```

## FPGA

Vivado 2022.2, ZCU102 (`xczu9eg-ffvb1156-2-e`).

| Configuration | Single-core | 16-core |
| --- | --- | --- |
| Design top | `wrapper1` | `wrapper16` |
| Simulation top | `tb_wrapper1` | `tb_wrapper16` |
| Synthesis | Vivado Synthesis Defaults | Vivado Synthesis Defaults |
| Implementation | Vivado Implementation Defaults | `Performance_ExplorePostRoutePhysOpt` |

Import `.v` files as design sources and `.sv` files as simulation sources. Read utilization and timing after implementation. The target clock period is 1.4 ns, achieving a maximum operating frequency of 714.3 MHz.
