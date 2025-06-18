# ⚡️ Electrical Blueprint Analyzer

A two-step pipeline to process electrical blueprints:

1. Detect, segments and return **wire electric length and map**
2. Extract **electrical symbols** (e.g., `100A`, `25A`) via OCR

---

## 🧩 Features

- ✅ Detect straight-line wires in architectural blueprints
- ✅ Calculate actual wire lengths return in pixels
- ✅ OCR key power symbols (e.g., ampere labels like `50A`, `150A`)
- ✅ Modular pipeline: each step is separate and reusable

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/trungthanhnguyenn/wire_extract.git
cd wire_extract
```

### 2. Create and activate environment

```bash
conda create -n ocr python=3.10
conda activate ocr
```

### 3. Install CUDA + PyTorch (CUDA 12.8)

```bash
conda install cuda -c nvidia/label/cuda-12.8.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 4. Install other dependencies

```bash
pip install -r requirements.txt
```

## 📂 Directory Structure (simplified)

```bash
project/
├── ocr/
│   ├── core/
│   │   ├── ocr_/                    # CRAFT + VietOCR modules
│   │   ├── deploy/
│   │   │   ├── extract_wire.py      # Step 1: Segmentation,detection and calculate wires length.
│   │   │   └── ocr_wire_symbol.py   # Step 2: OCR electrical symbols
│   └── weights/                     # Pretrained model weights
├── electric_line/                   # Tool to extract electric wire
├── .gitignore
└── README.md
```

## 🧪 How to Run

- From the project root:

```bash
PYTHONPATH=$(pwd) python ocr/core/deploy/extract_wire.py
PYTHONPATH=$(pwd) python ocr/core/deploy/ocr_wire_symbol.py
```

## 📁 Input & Output Setup
```markdown
Make sure you have your data prepared as follows in order to run:
- 1. extract_wire.py

| Variable       | Path                                     | Description                                             |
|----------------|------------------------------------------|---------------------------------------------------------|
| `electric_dir` | `/path/to/your/data/electric`            | Input folder with raw electrical diagram images         |
| `label_dir`    | `/path/to/your/data/label`               | Templates or patterns to detect wire endpoints          |
| `output_root`  | `/path/to/your/data//output`             | Output folder for all visualizations and results        |

- 2. ocr_wire_symbol.py
| Variable       | Path                                     | Description                                             |
|----------------|------------------------------------------|---------------------------------------------------------|
| `image_path`   | `/path/to/your/data/visual_result`       | Output image after extract wire process                 |
| `label_dir`    | `/path/to/your/data/label`               | Templates or patterns to detect wire endpoints          |
| `output_root`  | `/path/to/your/data//output`             | Output folder for visualizations and symbols OCR        |

> ⚠️ Ensure all folders exist and contain the required files before running the pipeline.
```

