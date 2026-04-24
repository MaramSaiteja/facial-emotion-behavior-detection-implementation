# Facial Emotion & Behavior Detection Implementation

A comprehensive deep learning project implementing multiple facial analysis components for emotion recognition, mask detection, and drowsiness detection using various frameworks.

## 🎯 Overview

This project brings together three sophisticated facial analysis components:
- **Emotion Detection** - Real-time facial emotion recognition
- **Mask Detection** - Detect whether people are wearing masks
- **Drowsiness Detection** - Monitor and detect driver drowsiness

## ✨ Components

### 1. Emotion Detection
Detects and classifies facial emotions in real-time using deep learning.

**Location:** `Emotion-Detection/`

**Features:**
- Real-time emotion recognition from webcam
- Support for training custom models
- Pre-trained model included (best accuracy)
- Displays emotion probabilities

**Quick Start:**
```bash
cd Emotion-Detection
pip install -r requirements.txt
python real_time_video.py
```

**Supported Emotions:** Happy, Sad, Angry, Surprised, Disgusted, Fearful, Neutral

---

### 2. Face Mask Detection
Detects faces and determines whether they are wearing masks.

**Location:** `Face-Mask-Detection/`

**Features:**
- Multi-framework support (PyTorch, TensorFlow, Keras, MXNet, Caffe)
- Lightweight model (1.01M parameters)
- 260x260 input size
- Optimized inference speed
- Web-based demo available

**Quick Start (PyTorch):**
```bash
cd Face-Mask-Detection
python pytorch_infer.py --img-path /path/to/image
```

**Web Demo:**
Visit [AIZOO Face Mask Detection](https://demo.aizoo.com/face-mask-detection.html) for an interactive demo.

---

### 3. Drowsiness Detection
Monitors and detects driver drowsiness for safety applications.

**Location:** `Drowsiness-Detection/`

**Features:**
- Real-time drowsiness monitoring
- Pre-trained models included
- Alert system with sounds

**Quick Start:**
```bash
cd Drowsiness-Detection
python detect_drowsiness.py
```

---

## 📋 Requirements

### Python Version
- Python 3.6+

### General Requirements
```bash
pip install opencv-python numpy tensorflow keras
```

For specific framework requirements, see individual component directories.

---

## 🚀 Project Structure

```
facial-emotion-behavior-detection-implementation/
├── Emotion-Detection/
│   ├── models/              # Pre-trained models
│   ├── haarcascade_files/   # Face detection cascade files
│   ├── fer2013/             # Dataset directory
│   ├── real_time_video.py   # Real-time detection script
│   ├── train_emotion_classifier.py
│   ├── load_and_process.py
│   ├── requirements.txt
│   └── README.md
├── Face-Mask-Detection/
│   ├── models/              # Pre-trained models
│   ├── load_model/          # Model loading utilities
│   ├── utils/               # Utility functions
│   ├── img/                 # Sample images
│   ├── pytorch_infer.py
│   ├── tensorflow_infer.py
│   ├── keras_infer.py
│   ├── mxnet_infer.py
│   ├── caffe_infer.py
│   └── README.md
└── Drowsiness-Detection/
    ├── model/               # Pre-trained models
    ├── sounds/              # Alert sounds
    ├── detect_drowsiness.py
    └── Exec Command.txt
```

---

## 📊 Model Architecture

### Emotion Detection
- Uses Convolutional Neural Networks (CNN)
- Trained on FER2013 dataset
- Outputs 7 emotion classes with probabilities

### Face Mask Detection
- Architecture: SSD (Single Shot MultiBox Detector)
- Backbone: Lightweight with 8 conv layers
- Total Parameters: 1.01M
- Input Size: 260x260
- Optimized for browser and edge devices

### Drowsiness Detection
- Eye aspect ratio-based detection
- Real-time monitoring capability
- Audio alerts for driver safety

---

## 🎓 Usage Examples

### Emotion Detection - Real-time
```bash
cd Emotion-Detection
python real_time_video.py
```

### Face Mask Detection - Image
```bash
cd Face-Mask-Detection
python pytorch_infer.py --img-path /path/to/image
```

### Face Mask Detection - Video/Webcam/surveillancecam
```bash
cd Face-Mask-Detection
# For video file
python pytorch_infer.py --img-mode 0 --video-path /path/to/video
# For webcam/surveillancecam
python pytorch_infer.py --img-mode 0 --video-path 0
```

### Drowsiness Detection
```bash
cd Drowsiness-Detection
python detect_drowsiness.py
```

---

## 📚 Framework Support

### Face Mask Detection Framework Options
- **PyTorch** - `pytorch_infer.py`
- **TensorFlow** - `tensorflow_infer.py`
- **Keras** - `keras_infer.py`
- **MXNet** - `mxnet_infer.py`
- **Caffe** - `caffe_infer.py` (requires caffe-ssd)

Simply replace `pytorch` with your preferred framework:
```bash
python tensorflow_infer.py --img-path /path/to/image
```

---

## 📦 Pre-trained Models

All components include pre-trained models:
- Emotion Detection: CNN model trained on FER2013
- Face Mask Detection: SSD model (multiple framework formats)
- Drowsiness Detection: Pre-trained detection model

**Note:** Download and place model files as indicated in individual component READMEs.

---


## 🤝 Attribution & Credits

### Original Project by Bhargav Chintam
This project is based on the original work by **Bhargav Chintam** , **Venkata Nagasai Teja Maram** , **Sumanth M** , **Lokireddy Sai Siddarth Readdy**:
- [GitHub Repository](https://github.com/bhargavchintam/FACIAL-EMOTION-BEHAVIOR-DETECTION-USING-DNN)

**Venkata Nagasai Teja Maram Co-Inventor:**
- Emotion Detection implementation and training
- Face Mask Detection models and inference code (multi-framework support)
- Drowsiness Detection implementation
- Project architecture and initial codebase

### Additional Contributions
- Repository consolidation and organization
- Documentation improvements
- Component integration

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🔗 References

- [Face Mask Detection Original](https://github.com/bhargavchintam/FACIAL-EMOTION-BEHAVIOR-DETECTION-USING-DNN/tree/master/Final%20Codes/Face%20Mask%20Detection)
- [Emotion Recognition Original](https://github.com/bhargavchintam/FACIAL-EMOTION-BEHAVIOR-DETECTION-USING-DNN/tree/master/Final%20Codes/Emotion%20Detection)
- [Drowsiness Detection Original](https://github.com/bhargavchintam/FACIAL-EMOTION-BEHAVIOR-DETECTION-USING-DNN/tree/master/Final%20Codes/Drowsiness%20Detection)
- [FER2013 Dataset](https://www.kaggle.com/c/3364)
- [AIZOO Face Mask Detection Demo](https://demo.aizoo.com/face-mask-detection.html)

---

## 🎯 Use Cases

- **Driver Safety:** Real-time drowsiness detection for driver monitoring
- **Security:** Mask detection for facility access control
- **Human-Computer Interaction:** Emotion detection for adaptive interfaces
- **Mental Health:** Emotion analysis for wellness applications
- **IoT/Edge Devices:** Lightweight models optimized for deployment

---

## ⚙️ System Requirements

- Python 3.6 or higher
- 4GB RAM minimum (8GB recommended)
- OpenCV-compatible system
- GPU support optional (CUDA for faster inference)

---

## 📝 Notes

- For face detection, OpenCV Haar Cascade files are included
- Model files are large; ensure sufficient storage space
- Caffe inference uses OpenCV DNN module or caffe-ssd fork
- Audio alerts in Drowsiness Detection require speaker/headphone setup

---

## 🤖 Contributing

This educational and research project consolidates implementations of facial emotion and behavior detection using deep neural networks. Components of this system were also implemented and tested within a university campus surveillance camera environment to evaluate real‑world monitoring scenarios.
- Report issues
- Suggest improvements
- Create pull requests
- Fork and adapt for your use cases

---

**Last Updated:** April 2026

For questions or issues, please refer to the original project or create an issue in this repository.
