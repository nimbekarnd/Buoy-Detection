# Buoy-Detection

## Problem Statement

The aim is to detect the buoys from the input video and using the proper color segmentation techniques detect the buoy.

[Link to Drive to access data and output along with codes](https://drive.google.com/drive/u/0/folders/1a6EyMh_ayfpkw_dAlJfLX-UkSbeZDIJI)

---

This involves implementing the concept of color segmentation using Gaussian Mixture Models and Expectation Maximization techniques.
- The input video sequence data has been captured underwater and shows three buoys of different colors, namely yellow, orange and green. They are almost circular in shape and are distinctly colored.
- Conventional segmentation techniques involving color thresholding will not work well in such an environment, since noise and varying light intensities will render any hard-coded thresholds ineffective. In such a scenario, the color distributions of the buoys is learnt and that learned model is used to segment them.
- A tight segmentation of each buoy for the entire video sequence has been obtained by applying a tight contour (in the respective color of the buoy being segmented) around each buoy.

### Approach and implementation:
- The data preparation phase involved computation and visualization of the average color histogram for each channel of the sampled RGB images for each colored buoy separately. This provides some intuition on how many
Gaussians (N) are required to fit to the color histogram.
- A 1-D Gaussian is used to model the color distribution of the buoys. Segmentation of the colored buoys was tried using this 1-D gaussian.
- Implementation of Expectation-Maximization algorithm was done and can be found in [sample_em_learning.py](https://github.com/nimbekarnd/Buoy-Detection/blob/main/Code/sample_em_learning.py).
- EM algorithm was used to compute the model parameters, i.e. the means and variances of the N 1-D Gaussians.
- Given the computed model parameters a color-segmented binary image was generated from the frames of the video sequence.
- Elaborate explanation about the approach and the pipeline can be found in the [report](https://github.com/nimbekarnd/Buoy-Detection/blob/main/Report.pdf)

### Output:

![All buoy detected](https://github.com/nimbekarnd/Buoy-Detection/blob/main/all_buoy_detected_gif.gif)   


**Output videos:**
- [Buoy detection/segmentation using single gaussian](https://drive.google.com/file/d/1GrSIRr0rLk_Xi-ZLIZP9X_oWDslYSiAZ/view).
- [Buoy detection/segmentation using multiple gaussians, here 3 for all buoys](https://drive.google.com/file/d/14QsOK3F9ndJLKVh7K_qXee6k0CbgWVAx/view?usp=sharing).

## Author
[Nupur Nimbekar](https://github.com/nimbekarnd)
