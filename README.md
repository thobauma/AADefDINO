# Defense in Self-Supervised Vision Transformers

Contents:

```
├── dino                        DINO git submodule
├── notebooks                   Various notebooks
│   └── 2012_2017_labels_map    ImageNet class mapping
├── scripts                     Various scripts
└── src                         Main code, supporting notebooks and scripts
    ├── helpers                 
    └── model
```

## Pipeline

### Adversarial Dataset Generation

This step prepares the dataset and generates the adversarial attacks (PGD, CW and FGSM) which are saved as tensors. 

- Download ImageNet train and validation set.
- `2012_2017_labels_map/`: notebook and mapping of the ImageNet classes from 2012 to 2017, with a notebook to generate those mappings contained in this folder.
- `notebooks/createDataSubset.ipynb`: we used a subset of 25 ImageNet classes. This notebooks creates the index for the subset.
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