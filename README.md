# EasyOCR Custom Model Training

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19K8gJyO5wkFuPt4kdB0jmfz12J6ZQWxN?usp=sharing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

A comprehensive guide to fine-tuning EasyOCR's text recognition models on custom datasets. This repository provides a step-by-step tutorial for training domain-specific OCR models that outperform generic pre-trained models on specialized text.

## Overview

While EasyOCR provides excellent out-of-the-box performance for general text recognition, it may struggle with:
- Technical jargon and domain-specific terminology
- Unique fonts or stylized text
- Specialized formatting (invoices, forms, technical documents)
- Low-quality or degraded text
- Languages or scripts with limited training data

This tutorial shows you how to fine-tune EasyOCR to achieve superior accuracy on your specific use case.

## Features

- Complete end-to-end training pipeline
- Automated data preprocessing and LMDB conversion
- Detailed explanations for each training step
- Model conversion utilities for EasyOCR compatibility
- Best practices and troubleshooting tips
- Ready-to-use Google Colab notebook

## Prerequisites

- Python 3.11 or higher
- Basic understanding of machine learning concepts
- Training images with corresponding text labels
- Google Colab account (for GPU training) or local GPU setup

## Quick Start

### Option 1: Google Colab (Recommended)

Click the badge below to open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/19K8gJyO5wkFuPt4kdB0jmfz12J6ZQWxN?usp=sharing)

### Option 2: Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/easyocr-custom-training.git
cd easyocr-custom-training
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Prepare your training data**
```
train_data/
├── image1.jpg
├── image2.jpg
├── ...
└── gt.txt
```

4. **Run the notebook**
```bash
jupyter notebook EasyOCR_Custom_Training.ipynb
```

## Repository Structure

```
easyocr-custom-training/
│
├── EasyOCR_Custom_Training.ipynb    # Main training notebook
├── README.md                         # This file
├── requirements.txt                  # Python dependencies
├── LICENSE                           # MIT License
│
├── examples/                         # Example data and results
│   ├── sample_data/                  # Sample training images
│   └── results/                      # Example outputs
│
└── utils/                            # Helper scripts (optional)
    ├── data_preparation.py
    └── model_converter.py
```

## Training Data Format

Your training data should follow this structure:

### Directory Structure
```
train_data/
├── image1.jpg
├── image2.jpg
├── image3.jpg
└── gt.txt
```

### Ground Truth File (`gt.txt`)

The ground truth file should contain tab-separated values:
```
image1.jpg	Hello World
image2.jpg	Sample Text
image3.jpg	Custom Label
```

**Important Notes:**
- Use **TAB** character (not spaces) between filename and label
- One line per image
- UTF-8 encoding
- Labels should match the exact text in the image

## Tutorial Contents

The notebook covers:

1. **Environment Setup**
   - Installing dependencies
   - Cloning the Deep Text Recognition Benchmark

2. **Data Preprocessing**
   - Ground truth file formatting
   - LMDB dataset creation

3. **Framework Compatibility**
   - PyTorch compatibility fixes
   - CPU/GPU configuration

4. **Model Training**
   - Architecture selection (VGG + BiLSTM + CTC)
   - Hyperparameter configuration
   - Training monitoring

5. **Model Conversion**
   - Converting to EasyOCR format
   - Model deployment preparation

6. **Testing & Evaluation**
   - Testing on sample images
   - Performance evaluation

## Configuration Options

### Model Architecture

Choose from different combinations:

| Component | Options |
|-----------|---------|
| **Transformation** | `None`, `TPS` |
| **Feature Extraction** | `VGG`, `RCNN`, `ResNet` |
| **Sequence Modeling** | `None`, `BiLSTM` |
| **Prediction** | `CTC`, `Attn` |

### Training Parameters

```python
--exp_name my_model          # Experiment name
--batch_size 8               # Batch size (adjust for GPU memory)
--num_iter 3000              # Total training iterations
--valInterval 100            # Validation frequency
--lr 1                       # Learning rate
--workers 4                  # Number of data loading workers
```

## Expected Results

With proper training data (200+ samples):

- **Training time**: 30-60 minutes (1000 iterations on GPU)
- **Accuracy improvement**: 20-50% over generic models
- **Best for**: Domain-specific text with 50+ unique vocabulary items

### Performance Metrics

Monitor these during training:
- **Train Loss**: Should steadily decrease
- **Validation Loss**: Should decrease without diverging from train loss
- **Accuracy**: Target 80%+ on validation set
- **Normalized Edit Distance**: Target < 0.10

## Troubleshooting

### Common Issues

**Q: Training loss not decreasing**
- Check data quality and label accuracy
- Increase training iterations
- Try different learning rates (0.5, 1.0, 2.0)

**Q: Out of memory errors**
- Reduce `batch_size`
- Use GPU runtime in Colab
- Reduce image resolution

**Q: Model not loading in EasyOCR**
- Verify model conversion completed successfully
- Check that converted model is in correct directory
- Ensure key names match EasyOCR's expected format

**Q: Low accuracy on validation set**
- Add more diverse training samples
- Increase `num_iter` to 3000-5000
- Try different model architectures

**Q: Overfitting (train accuracy >> validation accuracy)**
- Add more training data
- Reduce model complexity
- Implement data augmentation

## Tips for Better Results

### Data Collection
- Minimum 200 images recommended
- Cover all characters/symbols you need to recognize
- Include variations in lighting, angles, and quality
- Balance dataset (similar samples per class)

### Training Strategy
- Start with 1000 iterations, increase if needed
- Monitor validation metrics closely
- Save checkpoints regularly
- Test on completely unseen data

### Model Selection
- For **short text** (1-10 chars): VGG + BiLSTM + CTC
- For **longer text**: ResNet + BiLSTM + Attn
- For **simple fonts**: VGG + None + CTC (faster)

## Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Contribution Ideas
- Add example datasets for different domains
- Implement data augmentation utilities
- Create model evaluation scripts
- Add support for additional architectures
- Improve documentation

## Resources

- [EasyOCR Official Repository](https://github.com/JaidedAI/EasyOCR)
- [Deep Text Recognition Benchmark Paper](https://arxiv.org/abs/1904.01906)
- [CTC Loss Explanation](https://distill.pub/2017/ctc/)
- [LMDB Documentation](https://lmdb.readthedocs.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [JaidedAI](https://github.com/JaidedAI) for EasyOCR
- [Clova AI](https://github.com/clovaai) for Deep Text Recognition Benchmark
- The open-source OCR community

## Contact & Support

- **Issues**: [GitHub Issues](https://github.com/AbdullahButt2611/easyocr-custom-training/issues)
- **Discussions**: [GitHub Discussions](https://github.com/AbdullahButt2611/easyocr-custom-training/discussions)
- **Email**: abutt2210@gmail.com

## Star History

If you find this repository helpful, please consider giving it a star! It helps others discover this resource.

---

**Made with ❤️ for the OCR community**

*Last updated: January 2026*