# Alveolar canal 3D

3D neural network for the alveolar canal segmentation.
Model and set-up is implemented from [Jaskari et al.](https://www.nature.com/articles/s41598-020-62321-3.pdf)  
This project is related to our [recent work](#linkhere): our new 3D dense dataset can be downloaded [here](#linkhere).

## Usage
Once you have generated all the subsamples from our dataset you can run the experiments as follow:
```
usage: main.py [--base_config path]

optional arguments:
  --base_config        path to your config.yaml for this experimenti
  --verbose             redirect stream to std out instead of using a log file in the yaml directory
  --competitor          load training data as circle expansion instead of dense annotations
  --additional_dataset  load the additional patients
  --test                skip the training and load best weights for the experiment (no needs to update your yaml file)
  --skip_dump           if this flag is set the network does not dump prediction volumes on the final test
  --reload              load the last weights and continue the training (no needs to update your yaml file)
  --skip_primary        skip primary training test when loading data
```
Refer to the next sections for creating the circle expansion dataset and the subsamples needed for this repo.
## Path dataset

To generate the patch dataset:
```python
from utils import create_dataset
create_dataset(['train', 'syntetic', 'val', 'test'], is_competitor=True, saving_dir="/savedir/sparse")
create_dataset(['train', 'val', 'test'], is_competitor=False, saving_dir="/savedir/dense")
```

## Alpha shape
To create the alpha shape version of your patients refers to "alpha_shape.py" in this repo.

## Synthetic
to create the cyrcle-expanded dataset:
```python
from utils import create_syntetic
create_syntetic()
```


## YAML config example
Here is an example of a yaml file to use as base_config

```yaml
data-loader:
  augmentations_file: "path/to/augmentation.yaml"
  batch_size: 24
  file_path: "path/to/your/datadir"
  labels:
    BACKGROUND: 0
    INSIDE: 1
  num_workers: 4
loss:
  name: Jaccard
lr_scheduler:
  name: null
model:
  name: Competitor
optimizer:
  learning_rate: 0.0001
  name: Adam
seed: 47
tb_dir: "/path/to/tensorboard/dir"
title: ExperimentTitle
trainer:
  checkpoint_path: "pathToWeights.pth or None"
  do_train: true
  epochs: 100
```

In addiction we created a factory for Augmentation which allows you to load augmentations from a yaml file.
the following example can help you to make your own file. In our experiments we just used RandomFlip on all axes.

```yaml
RandomAffine:
  scales: !!python/tuple [0.8, 1.2]
  degrees: !!python/tuple [15, 15]
  isotropic: false
  image_interpolation: linear
  p: 0.35
RandomElasticDeformation:
    num_control_points: 7
    p: 0.35
RandomFlip:
  axes: 2
  flip_probability: 0.7
RandomBlur:
  p: 0.25
```

## Directories
Each experiment is expected to be placed into a result dir:

```
results/
├─ experiment_name/
│  ├─ checkpoints/
│  ├─ logs/
│  │  ├─ config.yaml
│  ├─ numpy/

```
If experiment_name does not exist, python will look for a *config.yaml* file in a *config* folder in your project directory.
