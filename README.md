# MSRGNN: Multi-Scale Relational Graph Neural Network for Unified Abstract Visual Reasoning

## Overview
MSRGNN is a unified model for solving various Abstract Visual Reasoning (AVR) tasks, consisting of a multi-scale panel-level feature extractor and a relational GNN reasoning module.

## Architecture
<img width="2364" height="2268" alt="main_figure" src="https://github.com/user-attachments/assets/2b6a952c-cac9-464f-8b00-94ac028026f5" />

## Project Structure
Each task is organised in its own self-contained folder
```
MSRGNN/
├── I_RAVEN # Contains MSRGNN and baselines for I-RAVEN experiments
├── RADIO # Contains MSRGNN and baselines for RADIO experiments
├── O3 # Contains MSRGNN and baselines for O3 experiments
├── viz # Contains visualisation code e.g. dataset distributions
├── datasets # Contains zip files of datasets
└── README.md   
```

## Installation

```bash
pip install -r requirements.txt
```
## Datasets
### I-RAVEN
I-RAVEN[1] can be found and generated from: https://github.com/cwhy/i-raven, but we provide our generated dataset in the "Releases" section at https://github.com/basiralab/MSRGNN. We do not include it directly in the repo due to size constraints.

### RADIO
RADIO, which is derived from OrganSMNIST [4,5] of MedMNIST [2,3], can be generated using the ```RADIO/gen_radio.py``` script provided and we provide our generated dataset at ```RADIO/generated_radio_datasets_split``` and in zip format at ```datasets/radio.zip```.

### O3 
O3[6] can be found at https://github.com/deepiq/deepiq, specifically the ```odd-one-out test examples.zip```. We also provide the dataset at   ```O3/odd-one-out``` and in zip format at ```datasets/odd-one-out test examples.zip```.

## Baseline Models
The state-of-the-art models which we evaluate against in our experiments include WReN [7], MXGNet [8], MRNet [9], DRNet [10] and SCAR [11] 

## Running experiments
Specific instructions for running experiments can be found in the README for each task's folder individually

## IMPORTANT - Pretrained SCAR and DRNet Model Weights
Due to GitHub limits on file size, SCAR model weights are saved in the "Releases" section at https://github.com/basiralab/MSRGNN. This is important as if you wish to use pre-trained weights you must manually place these in the correct directory.

For I-RAVEN, ```best_model_scar.pth``` and ```best_model_drnet.pth``` must be placed in ```I_RAVEN/saved_models```

For RADIO:
- ```RADIO1_SCAR```
- ```RADIO1_SCAR_TRANSFER_RAVEN```
- ```RADIO2_SCAR```
- ```RADIO2_SCAR_TRANSFER``` 
- ```RADIO1_DRNet```
- ```RADIO1_DRNet_TRANSFER_RAVEN```
- ```RADIO2_DRNet```
- ```RADIO2_DRNet_TRANSFER``` 

must be placed in ```RADIO/saved_models```

For O3:
- ```O3_5-SCAR```
- ```O3_5-SCAR-TRANSFER```
- ```O3_5-SCAR-TRANSFER_RADIO```
- ```O3_5-SCAR-TRANSFER_RADIO_2``` 

must be placed in ```O3/saved_models```

## Citations

[1] Hu at al. "Stratified Rule-Aware Network for Abstract Visual Reasoning" AAAI 2021.

[2] Yang, Jiancheng, Rui Shi, and Bingbing Ni. "MedMNIST Classification Decathlon: A Lightweight AutoML Benchmark for Medical Image Analysis." IEEE 18th International Symposium on Biomedical Imaging (ISBI), 2021, pp. 191-195.

[3] Yang, Jiancheng, Rui Shi, Donglai Wei, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, vol. 10, no. 1, 2023, p. 41.

[4] Bilic, Patrick, Patrick Ferdinand Christ, et al. "The Liver Tumor Segmentation Benchmark (LiTS)." CoRR, vol. abs/1901.04056, 2019.

[5] Xu, X., F. Zhou, et al. "Efficient Multiple Organ Localization in CT Image Using 3D Region Proposal Network." IEEE Transactions on Medical Imaging, vol. 38, no. 8, 2019, pp. 1885-1898.

[6] Mandziuk and Zychowski."DeepIQ: A Human-Inspired AI System for Solving IQ Test Problems" IJCNN 2019

[7] Barrett, Hill, Santoro et al. "Measuring abstract reasoning in neural networks." ICML 2018.

[8] Wang at al. "Abstract Diagrammatic Reasoning with Multiplex Graph Networks" ICLR 2020.

[9] Benny at al. "Scale-Localized Abstract Reasoning" CVPR 2021.

[10] Zhao et al. "Learning Visual Abstract Reasoning through Dual-Stream Networks" AAAI 2024

[11] Małkiński and Mańdziuk. "One self-configurable model to solve many abstract visual reasoning problems" AAAI 2024.
