# Ultrasound Nerve Segmentation
This project was done in preparation for my internship at Current Surgical.

## Project Overview
The objective of this project is to implement a segmentation model for Brachial Plexus (BP) nerves in ultrasound images and to obtain highest Dice score on test set as possible, where

$$\text{Dice Loss} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|}.$$

Here, $X$ represents the predicted mask and $Y$ represents the ground truth. Although Dice score might not be the best choice of metric in general for segmentation tasks, we have adopted it as the performance measure for practicing purposes. More details about the original project can be found on the [original Kaggle competition page](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation).

## Exploratory Data Analysis
During the exploratory data analysis phase, I focused on:
1. checking data imbalance.
2. examining conflicts in the ground truth masks.

For examining conflicts in the ground truth masks, we first pairwise compared the images and selected pairs where the images exhibited a significant resemblance (SSIM > 0.99). Then, any image whose mask does not contain nerves while its counterpart does was removed from the training set to resolve the conflict. In hindsight, since SSIM measure the "quality" of the images, it might not have been the most appropriate measure for this task. 

Also upon checking data imbalance, we discovered that approximately 40% of the images contained nerves and the size of the nerves (when present) accounted for approximately 3.2% of the image size on average. The percentage of non empty ground truth masks also varied greatly depending on the subject. 

To explore the appearances of the nerves and enhance their visibility, we tried two methods, histogram equalization and CLAHE, although neither of them was actually employed in the later modeling process. 

For more details on the exploratory data analysis, please refer to [exploratory.ipynb](exploratory.ipynb).

## Segmentation
Details of the model performance can be found in [Kaggle_score](Kaggle_scores.xlsx).

### Train/Validation set
For the train and test the model, 5-fold cross-validation was used. Each training, validation pair was splitted subjectwise and the ratio of the samples with ground truth masks in each set was kept at 39%~44%. There also was a local test set consisting of subjects 41-47 (again subjectwise disjoint from all training/validation sets), kept for tuning hyperparameters for ensembling methods. However, it was observed that there was a huge discrepancy between the models' performance on these subjects and the actual Kaggle set and for this reason, we ceased to use these for local testing. It was also observed that incorporating this test set back into training harmed models' peformance on Kaggle private LB. Combined with the aforementioned discrepancies of models' performance, we could only suspect that there is a significant difference between these test subjects and the actual Kaggle test set and these subjects were left out of training for the rest of the experiment.

### Loss functions/Metrics
The following loss functions were tested:

1. BCE Loss + Dice Loss (Dice Loss for saving),
2. Focal Loss + Dice Loss (Dice Loss for saving),
3. Wieghted Focal Loss + Dice Loss (Weighted Dice Loss ofr saving).

Focal Loss (weighted or unweighted) generated more/bigger masks (with Kaggle test set), however that did not necessarily translate to better private LB score. The combination of BCE Loss and Dice Loss was used for the rest of the experiment.

### Encoder/Decoder
Four ResNet encoders (ResNet10, ResNet18, ResNet34, ResNet50) were compared with ResNet34 outperforming the other 3.

For decoders, we compared UNet, UNet++, FPN, DeepLabV3+. We could achieve ~0.1 increase in the private LB score with DeepLabV3+. 

### Ensembling
Ensembling produced better scores than single models in general. For the final model the following ensembling method was used:

1. For each image, the threshold of 0.35 was used to produce predictions. 
2. Pixelwise average was taken from the predictions and was thresholded at 0.5 to generated a new prediction (i.e. voting).
3. If the resulting mask has size <500, then it was ignored.

Ensembling tends to reduce the mask size and the first step (of lowering the possibility threshold to 0.35) was added to counterbalance this. The final step was adopted as it was an easy way to filter out some false positive masks while minimising the loss coming from removing true positive masks. 

## Afterthoughts
A few things I could try:

1. Examining subject 41-47 more carefully. This might have helped us understanding the test set and possibly find more subject to include in/exclude from training.
2. Better pipeline for find tuning the parameters for ensembling. Since biggest improvement usually came from the "postprocessing" steps, it might be possible to increase the private LB score.
