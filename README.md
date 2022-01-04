# Attention for Adversarial Defense in Self-SupervisedVision Transformers

## Material

### Adversarial Dataset Generation

- Download ImageNet (see `setup/urls.txt`)
- `get_train_labels.ipynb`: map the labels
- `createDataSubset.ipynb`: define class subset
- contained in `emsemble.ipynb`: train classifier of head (25 classes)
- `scripts/adversarialDatasetGeneration.py`: generate all attacks (PGD, CW, FGSM)

### Posthoc

- `p_classifier_forward.ipynb`: Store the latent space for the posthoc classifier with a single forward pass. DISCUSS: for "n last 4 layers" latent space only and not for attention. ok like this?
- `p_classifier_train.ipynb`: trains the posthoc binary classifier.
- `p_classifier_matrix.ipynb`: computes the posthoc accuracy matrix

### Adversarial Training

- `adversarialTraining.ipynb`: performs adversarial training.

### Ensemble

- `emsemble.ipynb`: performs the ensemble defense

### Clustering Visualization

- `Clustering.ipynb`: performs the clustering of the latent space for post-hoc classifier as shown in the paper

### Rest

- `AdversarialBenchmark.ipynb`: launch attacks on DINO and generate accuracies for different attack parameters