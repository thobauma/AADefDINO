# Attention for Adversarial Defense in Self-SupervisedVision Transformers

## Material

### Adversarial Dataset Generation

- Download ImageNet (see `setup/urls.txt`)
- `get_train_labels.ipynb`: map the labels
- `createDataSubset.ipynb`: define class subset
- `file?`: train classifier of head (25 classes) (which file?)
- `scripts/adversarialDatasetGeneration.py`: generate all attacks (PGD, CW, FGSM)

### Posthoc

- `p_classifier_forward.ipynb`: Store the latent space for the posthoc classifier with a single forward pass.
- `p_classifier_train.ipynb`: trains the posthoc binary classifier.
- `p_classifier_matrix.ipynb`: computes the posthoc accuracy matrix

### Adversarial Training

- `file?`: performs adversarial training.

### Ensemble

- `emsemble.ipynb`: performs the ensemble defense

### Clustering Visualization

- `Clustering.ipynb`: 
- `Clustering-PostHoc.ipynb`: 
- `ViT_Output_Exploration.ipynb`: 