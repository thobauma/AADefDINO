# Adversarial Defense in Self-Supervised Vision Transformers

## Material

### Adversarial Dataset Generation

This step prepares the dataset and generates the adversarial attacks (PGD, CW and FGSM) which are saved as tensors. 

- Download ImageNet (see `setup/urls.txt`)
- `get_train_labels.ipynb`: map the labels
- `createDataSubset.ipynb`: define class subset
- contained in `emsemble.ipynb`: train classifier of head (25 classes)
- `scripts/adversarialDatasetGeneration.py`: generate all attacks (PGD, CW, FGSM)

### Adversarial Training

This step performs the adversarial training on the DINO classification head.

- `adversarialTraining.ipynb`: performs adversarial training.

### Posthoc classifier

This step generates and stores the latent space of DINO. This latent space is used as input to the post-hoc classifier.

- `p_classifier_forward.ipynb`: Store the latent space for the posthoc classifier with a single forward pass. DISCUSS: for "n last 4 layers" latent space only and not for attention. ok like this?
- `p_classifier_train.ipynb`: trains the posthoc binary classifier.
- `p_classifier_matrix.ipynb`: computes the posthoc accuracy matrix

This notebook was used to generate the visualizations of the latent space:

- `Clustering.ipynb`: performs the clustering of the latent space for post-hoc classifier as shown in the paper

### Ensemble

The ensemble model uses the posthoc classifier to defend against adversarial attacks.

- `emsemble.ipynb`: performs the ensemble defense

### Various

This notebook was used to calculate the accuracy of the (unmodified) DINO model against the considered adversarial attacks (PGD, CW, FGSM).

- `AdversarialBenchmark.ipynb`: launch attacks on DINO and generate accuracies for different attack parameters
