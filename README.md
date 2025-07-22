
# Visionary Palette: Image Colorization and Caption Generator 

This project presents a unified deep learning model capable of generating natural language captions for images and restoring colors to grayscale images. It leverages CNNs for feature extraction and Bi-LSTM networks for sequential caption generation, utilizing the Flickr30k dataset.

---

# Project Overview

- **Image Captioning**: Converts images into descriptive sentences using CNN + Bi-LSTM.
- **Image Colorization**: Adds realistic color to grayscale images using a CNN encoder-decoder.

The model shares CNN features across both tasks for better efficiency and learning.

---

# Dataset

**Flickr30k Dataset**
- 31,783 images
- 5 captions per image

| Split      | Images |
|------------|--------|
| Training   | 27,000 |
| Validation | 1,500  |
| Testing    | 1,500  |

---

# Model Architecture

# Image Captioning
- **Encoder**: Pre-trained CNN (e.g., VGG16)
- **Decoder**: Bi-LSTM
- **Loss**: Categorical Cross-Entropy
- **Metrics**: BLEU, METEOR

# Image Colorization
- **Color Space**: LAB (L - grayscale, A/B - color)
- **Architecture**: CNN encoder-decoder
- **Loss**: MSE (Pixel-wise), Perceptual Loss
- **Metrics**: PSNR, SSIM

---

# Setup Instructions

1. **Clone the Repository**
   
   git clone https://github.com/your-username/visionary-palette.git
   cd visionary-palette
   

2. **Install Requirements**
   
   pip install -r requirements.txt


3. **Prepare Dataset**
   - Download the [Flickr30k dataset](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset)
   - Place images and captions in the `data/` directory

4. **Run Training Scripts**
   
   python train_caption_model.py       # for captioning
   python train_colorization_model.py  # for colorization
   python unified_model.py             # for combined model
   

---

# Evaluation Metrics

 **Captioning**: BLEU-1 to BLEU-4, METEOR
 **Colorization**: PSNR (Peak Signal-to-Noise Ratio), SSIM (Structural Similarity Index)

---

# Results
 Image Captioning
 *Generated*: `two horses pull carriage driven by woman over snow covered ground`  
 *Ground Truth*: `Two draft horse pull a cart through the snow`

Image Colorization
- **PSNR** ≈ 35 dB → Excellent match
- **SSIM** ≈ 0.95 → Strong structural accuracy

---

Future Scope

- Improve diversity and contextual awareness in captions
- Incorporate multilingual & multi-modal input
- Enhance colorization for complex and abstract images
- Web app deployment using Flask/Streamlit

---
 License

This project is licensed under the [MIT License](LICENSE).


# References

1. Zhang et al., "Image Caption Generation Using Contextual Information Fusion With Bi-LSTM-s", IEEE Access, 2023.
2. Reddy et al., "Automated Image Colorization using Machine Learning", ICICT 2024.
3. Mitra et al., "Image Caption Generator Through Deep Learning", Springer, 2023.
4. Bhatt et al., "Deep Fusion: CNN-LSTM for Enhanced Captioning", CISCT 2023.
5. Nguyen et al., "Image Colorization Using Deep CNN", ASIAGRAPH 2016.
6. Zhang et al., "Image Colorization with CNN", CVPR 2016.



Acknowledgements

Developed as a final-year project under the Department of Information Technology,  
**SRM Valliammai Engineering College**, Chennai.
