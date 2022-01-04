# Defense in Self-Supervised Vision Transformers

Contents:

```
├── data_dir                    Directory in which all the data, models and labels are stored
├── dino                        DINO git submodule
├── notebooks                   Various notebooks
│   └── 2012_2017_labels_map    ImageNet class mapping
├── scripts                     Various scripts
└── src                         Main code, supporting notebooks and scripts
    ├── helpers                 
    └── model
```

## Pipeline

### Preparing ImageNet
- Download the train and the validation set of the ILSVRC2012 challenge from [ImageNet](https://image-net.org).
- Move all the training samples into `data_dir/ori/train/images`
- Move all the validation samples into `data_dir/ori/validation/images`

DINO used a different class mapping then the one in the ILSVRC2012 challenge.
The corrected labels are stored in `data_dir/ori/train/labels.csv` and `data_dir/ori/validation/labels.csv` respectively.
In order to map the ILSVRC2012 labels to the ones DINO used download the development kit ILSVRC2012_devkit_t12 from [ImageNet](https://image-net.org) and eecute the code in `notebooks/get_train_labels.ipynb` and `2012_2017_labels_map/`.
- `2012_2017_labels_map/`: notebook and mapping of the ImageNet classes from 2012 to 2017, with a notebook to generate those mappings contained in this folder.
- `notebooks/createDataSubset.ipynb`: we used a subset of 25 ImageNet classes. This notebooks creates the index for the subset. The created subset is stored in `data_dir/ori/class_subset.npy`

### Adversarial Dataset Generation

Here the adversarial attacks (PGD, CW and FGSM) are generated and saved as tensors. 

- `notebooks/emsemble.ipynb` (part): trains and saves the custom classifier head for our ImageNet subset. This is required since the classification head from the original DINO model is trained on the full ImageNet dataset which contains 1000 classes.
- `scripts/adversarialDatasetGeneration.py`: generates and saves adversarial dataset for PGD, CW and FGSM.
- `notebooks/AdversarialBenchmark.ipynb`: calculates the accuracy of DINO for the generated adversarial dataset.

### Adversarial Training

This step performs the adversarial training on the DINO classification head.

- `scripts/adversarialTraining.py`: performs adversarial on the classification head using the generated adversarial data.
- `notebooks/AdvTrainingMetrics.ipynb`: generates figure 2 (top-1 accuracy for 10 randomly selected classes after performing PGD adversarial trainingfor different values of ε)

### Post-hoc classifier

This step generates and stores the latent space of DINO. This latent space is used as input to the post-hoc classifier.

- `notebooks/p_classifier_forward.ipynb`: stores the latent space for the post-hoc classifier which is computed with a forward pass.
- `notebooks/p_classifier_train.ipynb`: trains the post-hoc binary linear classifier.
- `notebooks/p_classifier_matrix.ipynb`: computes the post-hoc accuracy matrix (table 4 in the report).
- `notebooks/Clustering.ipynb`: performs the clustering of the latent space for post-hoc classifier as shown in the paper

### Ensemble

The ensemble model uses the posthoc classifier to defend against adversarial attacks.

- `notebooks/emsemble.ipynb`: performs the ensemble defense
