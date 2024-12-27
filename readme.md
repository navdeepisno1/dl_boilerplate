|Loss type|Name|Usage|
|---------|-----|----|
|L1|Mean Absolute Error|Regression|
|||L2|Mean Squared Error|Regression|
|KLD|KL Divergence|Probabilty Distribution|
|CE|Cross Entropy|Classification|
|BCE|Binary Cross Entropy|Binary Classification|
|Dice|Dice|Segmentation|
|Huber|Huber|For Robust regression than MSE|
|CTC|Connectionist Temporal Classification Loss|For tasks where we need alignment between sequences, but where that alignment is difficult - e.g. aligning each character to its location in an audio file|
|SIOU|Shape Aware IOU|Used in bounding box regression, this loss function takes into account aspects like shape, distance, and aspect ratio alignment|
|Focal|Focal|This loss function helps fix the class imbalance problem by putting more focus on hard, misclassified examples|
|IOU|Intersection over Union|Object detection bbox|
|GIOU|Generalized IOU|Object detection bbox|
|Hungarian Matching|Hungarian Matching|Object detection bbox|
|Adversarial Loss|Adversarial Loss|GANs, Making Model Resistant to Adversarial Attacks, T2I Step Distillation, Privacy Preservation|
|Boundary Loss|Boundary Loss|Segmentation|
