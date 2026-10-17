# Decision Logic Interpretable Framework for Deepfake Detection
This repository contains scripts for training GenX-Net and DLIM, along with the corresponding data generators.
- **Trainer_1.py**
  The main training script for GenX-Net.
- **Image_preprocess_1.py**
  Implements the data generator for loading and preprocessing images used to train and validate GenX-Net.
  
- **Trainer_2.py**
  The main training script for DLIM.
  
- **Image_preprocess_2.py**
  Implements the data generator for loading and preprocessing images used to train and validate DLIM.
  
- **Setting.py**
  Contains dataset paths and hyperparameter configurations required for training.
  
## Requirements

- **Python**: 3.9 
- **TensorFlow**: 2.9.1  
- **Keras**: 2.9.0 (included in TensorFlow 2.9.1)  
- **NumPy**: 1.19.5  
- **Matplotlib**: 3.8.4  
- **OpenCV**: 4.12.0.88  
