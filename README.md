# Rice-STNet
## Official implementation of Rice-STNet for rice plant height estimation from multi-temporal UAV RGB imagery

---

## Overview

Rice-STNet is a spatiotemporal deep learning framework developed for accurate rice plant height estimation from multi-temporal UAV RGB imagery. Unlike conventional photogrammetry-based approaches that rely on explicit three-dimensional reconstruction, Rice-STNet directly learns the relationship between sequential canopy appearance and plant height through end-to-end spatiotemporal feature learning.

The repository accompanies the following manuscript:

> **Rice-STNet: A spatiotemporal deep learning framework for rice plant height estimation using multi-temporal UAV RGB imagery**

---

## Features

- End-to-end rice plant height estimation
- Multi-temporal UAV RGB image input
- CNN-based spatial feature extraction
- GRU-based temporal modeling
- Time2Vec temporal encoding
- Comparison with conventional machine learning methods
- High-resolution field-scale plant height mapping

---

## Repository Structure

```text
Rice-STNet
│
├── MachineLearningMethods/
│   ├── MakeDataInput.py
│   ├── RF.py
│   ├── SVR.py
│   └── commonFunction.py
│
├── RiceSTNet/
│   ├── dataset/
│   │   ├── dataset.json
│   │   ├── train.json
│   │   └── test.json
│   │
│   ├── RiceSTNet.py
│   ├── trainModel.py
│   ├── testModel.py
│   ├── Optimization.py
│   ├── makejson.py
│   ├── splitjs.py
│   ├── MSRCP.py
│   └── commonFunction.py
│
├── SimpleCNN/
│   ├── dataset/
│   ├── simpleCNN.py
│   ├── trainCNN.py
│   ├── testCNN.py
│   └── commonFunction.py
│
├── Visualization/
│   ├── HeatMap.py
│   ├── HeatMap_3D.py
│   ├── RiceSTNet.py
│   ├── MSRCP.py
│   ├── commonFunction.py
│   └── weights/
│
├── README.md
├── LICENSE
├── requirements.txt
└── .gitignore
```

---

## Requirements

Python 3.10 or later is recommended.

Main packages include

- PyTorch
- torchvision
- numpy
- pandas
- OpenCV
- rasterio
- matplotlib
- tqdm
- scikit-learn
- Optuna

Install all dependencies using

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

The original UAV imagery is **not included** in this repository because of its large storage size.

Users should organize the dataset according to the following structure:

```text
dataset/
│
├── RGB/
│
├── labels.csv
│
├── train.json
│
└── test.json
```

The provided JSON files illustrate the expected data organization.

---

## Training

Train the proposed Rice-STNet model using

```bash
python RiceSTNet/trainModel.py
```

---

## Model Evaluation

Evaluate the trained model using

```bash
python RiceSTNet/testModel.py
```

---

## Conventional Machine Learning Baselines

Random Forest

```bash
python MachineLearningMethods/RF.py
```

Support Vector Regression

```bash
python MachineLearningMethods/SVR.py
```

---

## Visualization

Generate field-scale plant height maps

```bash
python Visualization/HeatMap.py
```

Generate 3D visualization

```bash
python Visualization/HeatMap_3D.py
```

---

## Reproducibility

The random seeds used in this study are provided in the source code to facilitate reproducibility.

Model hyperparameters were optimized using Optuna with TPE-based Bayesian optimization and Hyperband pruning.

---

## Data Availability

Due to the large size of the UAV imagery and institutional data-sharing restrictions, the original datasets are not included in this repository.

The processed metadata and code are publicly available.

The original UAV images and associated annotations are available from the corresponding author upon reasonable request.

---

## Citation

If you use this code in your research, please cite

```bibtex
@article{wang2026ricestnet,
  title={Rice-STNet: A spatiotemporal deep learning framework for rice plant height estimation using multi-temporal UAV RGB imagery},
  author={Wang, Weiguo and ...},
  journal={XXXX},
  year={2026}
}
```

(The journal information can be updated after publication.)

---

## License

This project is released under the MIT License.

See the LICENSE file for details.

---

## Contact

**Weiguo Wang**

Graduate School of Agriculture

Hokkaido University

Sapporo, Japan

Email: your_email@xxx.jp
