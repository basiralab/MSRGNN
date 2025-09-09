# RADIO Task

## Project Structure

```
RADIO/
├── generated_radio_datasets_split/ # Contains the RADIO-1 and RADIO-2 dataset as ```.pkl``` files for train, val and test splits
├── run/s # Contains tensorboard training logs for baseline models and for the final proposed MSRGNN model
├── saved_models/ # Contains saved model ```.pth``` files for all models
└── README.md  
 ```

### Dataset generation code
```gen_radio.py``` contains code to generate the datasets for RADIO-1 and RADIO-2 which can be run using ```python gen_radio.py```
This creates the generated_radio_datasets_split folder containing ```.pkl``` files for train, val and test splits for each dataset

### Data utilities
```radio_data_utility``` Provides data utility and loading for the RADIO datasets

### Models
The models can be found in the following files:
- ```msrgnn.py```  # Our proposed MSRGNN
- ```drnet.py```   # From https://github.com/VecchioID/DRNet
- ```scar.py```    # From https://github.com/mikomel/sal
- ```mxgnet.py```  # From https://github.com/thematrixduo/MXGNet
- ```mrnet.py```   # From https://github.com/yanivbenny/MRNet
- ```wren.py```    # From https://github.com/Fen9/WReN

### Jupyter Notebooks
All experiments can be run using the relevant notebook files:

```RADIO_1_<model>.ipynb``` for the RADIO-1 experiments which performs nested cross-validation, followed by training a final model for transfer in the RADIO-2 task, as well as transfer learning task from I-RAVEN to RADIO-1

```RADIO_2_<model>.ipynb``` for the RADIO-2 experiments which performs nested cross-validation, followed by fine-tuning a model from the trained model from RADIO-1 for transfer learning

All relevant measures are then stored by tensorboard and can be accessed with: ```tensorboard --logdir=runs``` from the specific directory e.g. from ```MSRGNN/RADIO```.

The trained best models for each fold, and the final models for transfer learning can be found in the ```saved_models``` folder.


