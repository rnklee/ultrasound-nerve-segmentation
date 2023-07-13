# Ultrasound Nerve Segmentation
This project was done in preparation for my upcoming internship at Current Surgical.

## Project Overview
The objective of this project is to implement a segmentation model that correctly detects and locates Brachial Plexus (BP) nerves in ultrasound images. The performance of the model is measured by the Dice loss:

$$\text{Dice Loss} = \frac{2 \cdot |X \cap Y|}{|X| + |Y|}.$$

Here, $X$ represents the predicted mask and $Y$ represents the ground truth. Although Dice loss might not be the best choice of metric in general for segmentation tasks, I have adopted it as the performance measure for practicing purposes. More details about the project can be found on the [original Kaggle competition page](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation).

## Exploratory Data Analysis
During the exploratory data analysis phase, I focused on the following tasks:
1. examining and resolving conflicts in the ground truth masks,
2. exploring the appearances of the nerves and improving their visibility,
3. checking data imbalance.

For more details on the exploratory data analysis, please refer to the [nerve_segmentation_exploratory.ipynb](nerve_segmentation_exploratory.ipynb) notebook.

## Segmentation Model
This part is currently undergoing some edits, and more explanations will be added soon.
