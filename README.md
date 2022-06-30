# Defense in Self-Supervised Vision Transformers

Contents:

```
├── data                    Directory in which all the data, models and labels are stored
├── dino                        DINO git submodule
├── notebooks                   Various notebooks
│   └── 2012_2017_labels_map    ImageNet class mapping
├── converted_scripts           Most of the experiments
└── src                         Main code, supporting notebooks and scripts
    ├── helpers                 
    └── model
```

## Pipeline

### Preparing ImageNet
- Download the train and the validation set of the ILSVRC2012 challenge from [ImageNet](https://image-net.org).
- Move all the training samples into `data/ori/train/images`
- Move all the validation samples into `data/ori/validation/images`
- Create the filtered dataset with `notebooks/SuperclassingImageNet.ipynb` and move it into `data/ori/filtered/train/images` and `data/ori/filtered/validation/images` respectively.

### Clean linear classifier head
- Train a linear classifier on the filtered dataset: `converted_scripts/train_linear_classifier.py`.

### Adversarial Dataset Generation

Here the adversarial attacks (PGD, CW and FGSM) are generated and saved as tensors. 
- `converted_scripts/adversarialDatasetGeneration.py`: generates and saves adversarial dataset for PGD, CW and FGSM.
- `converted_scripts/AdversarialBenchmark.py`: calculates the accuracy of DINO for the generated adversarial dataset.

### Adversarial Training
This step performs the adversarial training on the DINO classification head.
- `converted_scripts/adversarialTraining.py`: performs adversarial on the classification head using the generated adversarial data.
- `notebooks/AdvTrainingMetrics.ipynb`: generates figure 2 (top-1 accuracy for 10 randomly selected classes after performing PGD adversarial trainingfor different values of ε)

### Post-hoc classifier
This step generates and stores the latent space of DINO. This latent space is used as input to the post-hoc classifier.
- `converted_scripts/p_classifier_train.py`: trains the post-hoc binary linear classifier on the filtered data together with the different adversarial pgd attacks.
- `converted_scripts/p_classifier_matrix.py`: computes the post-hoc accuracy matrix (table 7 in the report).
- `notebooks/Clustering.ipynb`: performs the clustering of the latent space for post-hoc classifier as shown in the paper.

### Ensemble

The ensemble model uses the posthoc classifier to defend against adversarial attacks.

- `notebooks/emsemble.ipynb`: performs the ensemble defense
