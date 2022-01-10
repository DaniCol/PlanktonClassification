# challenge kaggle

Deep plankton classification - Kaggle Challenge. This challenge is proposed in the context of the Deep learning lecture.

## Preprocessing

Different option are available for preprocessing our data

- Resizing stategy
  - Square Padding
  - Resize and Crop
  - Resize (lose ratio)
Only one has to be activated. See <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/challenge-kaggle/-/blob/master/src/config.yaml#L5" title="load_model">[here]</a>

- Reverse Color
  - True: Black background and white image
  - False: White background and black image
- Normalize
  - If you want to normalize your images
- Data Augmentation
  - Horizontal Flip
  - Vertical Flip
  - Affine

## Choose or add a new model

In the configuration file you can choose :
```
    MODEL : 'LinearNet'
    MODEL : 'ConvNet'
```
To create a new model, save it in `src/models`
If you want to add a new model, go to <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/challenge-kaggle/-/blob/master/src/tools/utils.py#L22" title="load_model">[here]</a> and add a `if/elif` statement.

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
