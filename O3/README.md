# O3 Task

## Project Structure

```
O3/
├── odd-one-out/ # Contains the O3 dataset where each problem instance is an image of all panels
├── runs/ # Contains tensorboard training logs for baseline models and for the final proposed MSRGNN model
├── saved_models/ # Contains saved model ```.pth``` files for all models
├── data_utility.py # Contains the data loader for the O3 dataset
└── README.md
 ```

### Models
The models can be found in the following files:
- ```msrgnn.py``` # Our proposed MSRGNN model
- ```scar.py```   # From https://github.com/mikomel/sal
- ```wren.py```   # From https://github.com/Fen9/WReN

### Jupyter Notebooks
All experiments can be run using the relevant notebook files:

```O3_<model>_5.ipynb``` for the O3 experiments which performs nested cross-validation for STL and also TL tasks

All relevant measures are then stored by tensorboard and can be accessed with: ```tensorboard --logdir=runs``` from the specific directory e.g. from ```MSRGNN/O3```.