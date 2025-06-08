# Model Documentation

This document provides detailed information about the Sybil model architecture, training process, and prediction pipeline.

## Model Architecture

Sybil is a deep learning model designed for lung cancer risk prediction from chest CT scans. The model uses an ensemble approach with multiple checkpoints for robust predictions.

### Key Components

1. **Base Model**
   - Deep neural network architecture
   - Attention mechanism for focusing on relevant regions
   - Ensemble of 5 model checkpoints

2. **Calibrator**
   - Simple calibrator for probability calibration
   - Improves prediction reliability
   - Stored in JSON format

## Model Checkpoints

The model uses 5 different checkpoints for ensemble prediction:

1. `28a7cd44f5bcd3e6cc760b65c7e0d54d.ckpt`
2. `56ce1a7d241dc342982f5466c4a9d7ef.ckpt`
3. `64a91b25f84141d32852e75a3aec7305.ckpt`
4. `65fd1f04cb4c5847d86a9ed8ba31ac1a.ckpt`
5. `624407ef8e3a2a009f9fa51f9846fe9a.ckpt`

## Prediction Pipeline

### 1. Input Processing

1. **File Loading**
   - Supports DICOM and PNG formats
   - Handles multiple files in a series
   - Maintains proper ordering

2. **Preprocessing**
   - Image normalization
   - Voxel spacing adjustment
   - Metadata preservation

### 2. Model Prediction

1. **Ensemble Prediction**
   - Parallel processing of multiple checkpoints
   - Weighted averaging of predictions
   - Confidence scoring

2. **Attention Visualization**
   - Heatmap generation
   - Overlay creation
   - GIF animation support

### 3. Output Generation

1. **Prediction Scores**
   - Risk assessment
   - Confidence intervals
   - Attention scores

2. **Visualization**
   - Overlay images
   - Attention maps
   - Animated GIFs

## Model Configuration

The model can be configured using the following parameters:

```python
MODEL_CONFIG = {
    "RETURN_ATTENTIONS_DEFAULT": True,
    "WRITE_ATTENTION_IMAGES_DEFAULT": True,
    "SAVE_AS_DICOM_DEFAULT": True,
    "SAVE_ORIGINAL_DEFAULT": True
}
```

## Performance Considerations

1. **GPU Acceleration**
   - CUDA support for faster processing
   - Multi-threading for parallel operations
   - Memory optimization

2. **Resource Usage**
   - Efficient memory management
   - Batch processing capabilities
   - Temporary file cleanup

## Model Updates

The model can be updated by:

1. Replacing checkpoint files
2. Updating the calibrator
3. Modifying configuration parameters

## Citation

If you use this model in your research, please cite:

```bibtex
@article{mikhael2023sybil,
  title={Sybil: a validated deep learning model to predict future lung cancer risk from a single low-dose chest computed tomography},
  author={Mikhael, Peter G and Wohlwend, Jeremy and Yala, Adam and Karstens, Ludvig and Xiang, Justin and Takigami, Angelo K and Bourgouin, Patrick P and Chan, PuiYee and Mrah, Sofiane and Amayri, Wael and Juan, Yu-Hsiang and Yang, Cheng-Ta and Wan, Yung-Liang and Lin, Gigin and Sequist, Lecia V and Fintelmann, Florian J. and Barzilay, Regina},
  journal={Journal of Clinical Oncology},
  pages={JCO--22},
  year={2023},
  publisher={Wolters Kluwer Health}
}
```
