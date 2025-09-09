# I-RAVEN Task

## Project Structure

```
I_RAVEN/
├── ablations/ # Contains experiments for MSRGNN ablation experiments in a self-contained folder
├── data_utility/ # Contains data loaders for all models
├── runs/ # Contains tensorboard training logs for baseline models and for the final proposed MSRGNN model
├── saved_models/ # Contains saved model ```.pth``` files for all models
└── README.md
 ```

### Models
The model files contain the model, followed by a main function which runs the relevant training and testing script.

The models can be found in the following files:
- ```msrgnn.py```  # Our proposed MSRGNN
- ```drnet.py```   # From https://github.com/VecchioID/DRNet
- ```scar.py```    # From https://github.com/mikomel/sal
- ```mxgnet.py```  # From https://github.com/thematrixduo/MXGNet
- ```mrnet.py```   # From https://github.com/yanivbenny/MRNet
- ```wren.py```    # From https://github.com/Fen9/WReN

### Dataset
To load the dataset from the pre-generated files, you must first unzip the file found in ```MSRGNN/datasets/I-RAVEN```, the unzipped result should then be stored such that it can be later used when running the model by passing the path as a command-line argument

### Data utilities
```data_utility_norm.py``` - Provides a data loader which returns normalised tensors in the range of [0, 1] used for most models

```data_utility_alb.py``` - Provides a data loader specifically for SCAR which uses custom albumentations as described in its paper

```data_utility.py``` - Provides an unnormalised data loader for models which have their own model-specific normalisation logic built into their provided model code


All models can be run using ```python <model> --data <dataset_path>``` from this directory, which performs full training followed by testing.

All relevant measures are then stored by tensorboard and can be accessed with: ```tensorboard --logdir=runs``` from the specific directory e.g. from ```MSRGNN/I-RAVEN```.

The trained best model will then be saved as ```saved_models/best_model_<model_name>.pth```.

Ablations can be run by following the above steps, but from within the ablations directory.

