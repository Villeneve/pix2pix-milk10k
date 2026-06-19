# Pix2Pix with VGG16 Feature Reconstruction Loss

We trained a Pix2Pix model using perceptual reconstruction losses derived from intermediate layers of a pretrained VGG16 network. The objective was to minimize the Mean Absolute Error (MAE) between feature maps extracted from generated and target images.

Each experiment used a specific VGG16 layer, denoted as **Lx**, where *x* corresponds to the selected VGG16 layer. Experiments prefixed with **10Lx** indicate that the perceptual reconstruction loss received a weight of **10×** relative to the adversarial loss during training.

The results show that deeper perceptual features can significantly impact image quality. In particular, **L9** achieved the best Fréchet Inception Distance (FID), suggesting that intermediate-to-deep feature representations provide a favorable balance between structural fidelity and perceptual realism.

## Results

| Configuration | Description | FID |
|-------------|-------------|------:|
| 10MAE | Pixel-wise MAE reconstruction loss (10× adversarial weight) | 82.50 |
| L2 | VGG16 feature reconstruction at layer 2 | 69.74 |
| 10L2 | VGG16 feature reconstruction at layer 2 (10× weight) | 67.47 |
| L9 | VGG16 feature reconstruction at layer 9 | 86.35 |
| 10L9 | VGG16 feature reconstruction at layer 9 (10× weight) | 75.06 |
| L16 | VGG16 feature reconstruction at layer 16 | 77.55 |
| 10L16 | VGG16 feature reconstruction at layer 16 (10× weight) | 76.17 |

## Discussion

Applying perceptual reconstruction losses generally improved the quality of the generated images compared to a heavily weighted pixel-wise MAE objective. The best result was obtained with **10L2**, achieving an FID of **67.47**, indicating that emphasizing low-level VGG16 features while maintaining adversarial supervision can lead to more realistic image synthesis.

Increasing the perceptual loss weight from 1× to 10× improved performance for both **L2** and **L9**, while only marginally affecting **L16**. This suggests that lower and intermediate VGG16 features provide stronger guidance for the Pix2Pix generator in this task than deeper semantic representations.
