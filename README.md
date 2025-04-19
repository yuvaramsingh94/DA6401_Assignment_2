# DA6401_Assignment_2


Author : V M Vijaya Yuvaram Singh (DA24S015)

## Report
[Wandb report](https://wandb.ai/yuvaramsingh/lightning_test/reports/DA6401-Assignment-2-Yuvaram--VmlldzoxMjMzOTk2OA)

## Github
[Link](https://github.com/yuvaramsingh94/DA6401_Assignment_2)

## Folder organization
```
DA6401_Assignment_2/
|   |── Part_a \\ codes for Part A question
|       |── CNNNetwork.py
|       |── config.py \\ Configuration of the best scratch model
|       |── dataloader.py
|       |── lightning.ipynb
|       |── LightningModule.py   
|       |── predictions.py
|       |── sweep.py \\ Code to perform hyperparameter sweep
|       |── train.py \\ Code to train the model
|       |── utils.py
|   |── Part_b \\ codes for Part B question
|       |── CNNNetwork.py
|       |── config.py
|       |── dataloader.py
|       |── lightning.ipynb
|       |── LightningModule.py   
|       |── predictions.py
|       |── train.py
|       |── utils.py
|   |── dataset
|   |── weights
|   |── requirments.txt 
```

## How to run
### Part A
#### Running hyperparameter tuning
```
python Part_a/sweep.py
```
#### Training the best model
Edit the configuration file if required
```
python Part_a/train.py
```
### Part B
#### Training the best model
Edit the configuration file if required
```
python Part_b/train.py
```
