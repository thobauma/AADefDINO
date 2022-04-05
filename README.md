# ToDo
- [ ] Report
    - [ ] fix the structure of the report (nasib, javi) (19.04.2022)
    - [ ] some more future work (nasib) (19.04.2022)
    - [ ] formatting for ICML (max) (19.04.2022)
- [ ] Codebase and experiments
    - [ ] Define additional results that are needed (Javi) (19.04.2022)
    - [ ] {BACKLOG} compare the post hoc on dino with a model directly trained on the original and adversarial images (not sure? postpone or clarify)
        - [ ] maybe get resnet feature extractor, train classifier (post-hoc and two classification heads)
    - [ ] (if above completed) Generate more results (all)
        - [ ] different pgd parameters (more in depth)
    - [ ] change jupyterfiles to proper python (Thomas, Max) (19.04.2022)
    - [ ] {BACKLOG} DINO pretrained on imagenet (might be more robust to that dataset) --> finetune on seperate dataset and see how well it performs against adversarial attacks (does it generalise)
- [ ] General
    - [x] Ask Luis for feedback&Ideas (Javi) (19.04.2022)
    - [ ] Go through the paper, write down ideas and things to be improved, share in group (All) (19.04.2022)
    - [ ] Look at a few papers that were accepted as NIPS workshops (All) (19.04.2022)
    - [ ] Two approaches
        - [ ] Try to reformulate so that we are exploring how these attacks are affecting DINO and draw conclusion
        - [ ] Try to show that we can actually build a defence without retraining entire transformer with adversarial training (not sure if feasible)
    

# Feedback on project condensed
- Structure-wise I find it a bit odd that the Limitations section comes last in the paper and some more discussion of possible future directions would have been nice. --> maybe ignore the limitations comment.
- it would be better if the authors compare their post-hoc detector with a model directly trained on original and adversarial images. It is unclear why the authors choose the detector to be post-hoc, as the authors only mentions its downside in Sec 2.2.1. --> highlight post-hoc benefits
- The fact that they only focus on fine-tuning/defending in the last layer of a transformer is very interesting. Even though the authors mention that this is a limitation of their results, I think this could have been argued for in a more positive manner: it seems like a very interesting approach to me, since for large pre-trained models it may not be feasible to defend on all layers, even when much more compute is available. --> maybe we did not justify our approach, which we took for granted or did not stress enough - clarify why post-hoc as defence rather full scale defence


## Deadline NIPS:   TBD for workshops (after paper deadline), May 16th for abstract, May 19th for paper

## Deadline ICML AdvML Workshop:  May 23rd for workshop submission ==> attempt this
https://advml-frontier.github.io/
(https://openreview.net/)

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
