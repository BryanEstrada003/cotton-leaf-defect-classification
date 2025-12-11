# cotton-leaf-defect-classification

note: crear an environment

pip install -r requirements.txt

## Proposed Methodology
1. Select one of the best model KAN (maybe if model had the best performance or it is short reference in labels)
2. train this model with the dataset of imagenet (using tensorflow) (hyperparameters using x default lr=1e-3, adam optimization, batch 64/32, 100 epoch, 0.5 dropout)
3. save the weights (weight matrix) and the architecture with the weights
4. training CNN model (vgg16) and KAN model with cotton leaf defect dataset (hyperparameters maybe using x default but if we do not best performance using bayessian optimization (just only: lr, l2 regularization, dropout)) (using training and validation set)
5. evaluate with test set using classical metrics (accuracy, recall, f1-score, precision)
