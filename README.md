# Challenge kaggle

Deep plankton classification - Kaggle Challenge. This challenge is proposed in the context of the Deep learning lecture.
You can find our recorded presentation on Youtube <a href="https://www.youtube.com/watch?v=KZ79-MmqmdI" title="youtube_video">[here]</a>

## Launch the training

```bash
cd ./src
python3 train.py --path_to_config ./config.yaml
```

## Track the training with tensorboard

```bash
cd ./src
tensorboard --logdir ./tensorboard/
```

## Launch inference on the test set

```bash
cd ./src
python3 inference.py --path_to_config ./config.yaml
```

## Launch model averaging on the test set

```bash
cd ./src
python3 average_inference.py --path_to_config ./config.yaml
```

## Model averaging

In the configuration file, activate Average:

```yaml
TEST:
  BATCH_SIZE: 128
  PATH_TO_MODEL: '../models/LinearNet_0/best_model.pth'
  PATH_TO_CSV: './test.csv'
  AVERAGE:
    ACTIVE: True
    PATH:
      - {MODEL: '../models/linearnet_0/best_model.pth', CONFIG: '../models/linearnet_0/config_file.yaml'}
      - {MODEL: '../models/linearnet_1/best_model.pth', CONFIG: '../models/linearnet_1/config_file.yaml'}

```

You must fill in all the paths of the configuration files of the models you want to average.
It will create temporary csv files containing the probabilities of each class per image per model.
The output file containing the predictions of the final classes has the name `cfg['TEST']['PATH_TO_CSV']`

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

```yaml
    MODEL : 'LinearNet'
    MODEL : 'ConvNet'
```

To create a new model, save it in `src/models`
If you want to add a new model, go to <a href="https://gitlab-student.centralesupelec.fr/2018barreeg/challenge-kaggle/-/blob/master/src/tools/utils.py#L22" title="load_model">[here]</a> and add a `if/elif` statement.

## Connect to the cluster

Documentation dcejs : https://dce.pages.centralesupelec.fr/

Reservation :

```bash
Without reservation
gpu_prod_long
walltime 08:00
```

Batch training tutorial :
https://dce.pages.centralesupelec.fr/05_examples/#a-more-advanced-sbatch

## Run en batch

Se connecter en ssh et entrer le password.
Créer un dossier
Ex :

```bash
ssh gpusdi1_21@chome.metz.supelec.fr
gpu2020sdi1
cd "TON dossier oU YA le code"
mkdir -p logslurms
touch job.batch
vim job.batch

COPY n PASTE ça :

#!/bin/bash

#SBATCH --job-name=emnist
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod
#SBATCH --time=24:00:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err

mkdir -p logslurms
python3 train.py --path_to_config

Ferme et save.
Ensuite :

sbatch job.batch

puis liste les jobs pour voir si il tourne

squeue
squeue -u <ton_id>
```
