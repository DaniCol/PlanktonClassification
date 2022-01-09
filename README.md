# challenge kaggle

Deep plankton classification - Kaggle Challenge. This challenge is proposed in the context of the Deep learning lecture.


## Launch the training

```
cd ./src
python3 train.py --path_to_config ./config.yaml
```

## Track the training with tensorboard

```
cd ./src
tensorboard --logdir ./tensorboard/
```

## Launch inference on the test set

```
cd ./src
python3 inference.py --path_to_config ./config.yaml
```

## Connect to the cluster

Documentation dcejs : https://dce.pages.centralesupelec.fr/

Reservation :
```
Without reservation
gpu_prod_long
walltime 08:00
```

Batch training tutorial :
https://dce.pages.centralesupelec.fr/05_examples/#a-more-advanced-sbatch