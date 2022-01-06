# challenge kaggle

Deep plankton classification - Kaggle Challenge. This challenge is proposed in the context of the Deep learning lecture.


## Launch the training

```
cd ./src
python3 train.py --path_to_config ./config.yml
```

## Track the training with tensorboard

```
cd ./src
tensorboard --logdir ./tensorboard/
```

## Launch inference on the test set

```
cd ./src
python3 inference.py --path_to_config ./config.yml
```
