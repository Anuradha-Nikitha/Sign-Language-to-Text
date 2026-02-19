# 🖐️ Sign Language to Text & Speech System

A real-time sign language recognition system that converts hand gestures (A-Z alphabets) into text and speech using deep learning and computer vision.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📖 Overview

This project is an end-to-end AI-based application that recognizes American Sign Language (ASL) alphabets in real-time and converts them into:
- ✅ **Text** (Words & Sentences)
- ✅ **Speech** (Audio Output via Text-to-Speech)

The system uses:
- **OpenCV** for real-time hand gesture capture and preprocessing
- **Convolutional Neural Networks (CNN)** for alphabet classification (A-Z)
- **Custom Sentence Builder** for word and sentence formation
- **pyttsx3** for text-to-speech conversion

---

## 🎯 Key Features

- 📷 **Real-time Webcam Detection** - Captures and processes hand gestures live
- 🧠 **Deep Learning Classification** - CNN model trained to recognize 26 alphabets (A-Z, excluding J which requires motion)
- 🔤 **Word & Sentence Formation** - Build complete sentences letter by letter
- 🔊 **Speech Synthesis** - Convert recognized text to speech
- 💾 **Audio Download** - Save generated speech as `.wav` file
- 🖥️ **Interactive UI** - Clean interface with real-time feedback
- 🔁 **Dual Hand Support** - Works with both left and right hands
- ⚡ **High Accuracy** - Confidence threshold filtering for reliable predictions

---

## 🏗️ Project Structure

```
Sign-Language-to-Text-master/
├── final_opencv_app.py          # Main application (GUI + Real-time Recognition)
├── train_final_alphabets.py     # Training script for CNN model
├── test_final_alphabet.py       # Testing script with speech output
├── collect_alphabets.py         # Data collection tool
├── requirements_pip.txt         # Python dependencies
│
├── final_data/                  # Training dataset
│   └── train/
│       ├── A/
│       ├── B/
│       ├── C/
│       └── ... (Z)
│
├── final_models/                # Trained models
│   └── alphabets/
│       ├── model.h5             # Keras model
│       └── labels.txt           # Class labels
│
├── backup_models/               # Backup models and training data
│   ├── models_backup/
│   └── data/
│
├── sentence/                    # Sentence building module
│   └── sentence.py              # SentenceBuilder class
│
└── README.md                    # This file
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows/Linux/MacOS

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/Sign-Language-to-Text.git
cd Sign-Language-to-Text-master
```

### Step 2: Install Dependencies
```bash
pip install -r requirements_pip.txt
```

**Key Dependencies:**
- TensorFlow 2.10.0
- OpenCV 4.7.0.72
- NumPy 1.23.5
- pyttsx3 2.90 (Text-to-Speech)
- Keras 2.10.0

---

## 📚 Usage

### 1️⃣ Data Collection (Optional - if training new model)

Use this script to collect hand gesture images for training:

```bash
python collect_alphabets.py
```

**How it works:**
- Shows webcam feed with a green ROI (Region of Interest) box
- Press the letter key (A-Z) to capture images for that alphabet
- Collects up to 300 images per letter
- Press `ESC` to exit

**Tips:**
- Use good lighting
- Vary hand positions and angles
- Capture with both hands for better model generalization

---

### 2️⃣ Training the Model

Train the CNN model on your collected dataset:

```bash
python train_final_alphabets.py
```

**Training Details:**
- **Input Size:** 128x128 grayscale images
- **Architecture:** 3 Conv2D layers + MaxPooling + Dense layers
- **Epochs:** 20 (configurable)
- **Batch Size:** 32
- **Augmentation:** Rotation, zoom, shift, horizontal flip (for left/right hand support)
- **Output:** Saves model to `final_models/alphabets/model.h5`

---

### 3️⃣ Testing the Model

Test the trained model with simple UI:

```bash
python test_final_alphabet.py
```

**Controls:**
- `C` - Confirm detected letter
- `S` - Add space (complete word)
- `V` - Speak the sentence
- `ESC` - Exit

---

### 4️⃣ Running the Final Application

Launch the full-featured application with advanced UI:

```bash
python final_opencv_app.py
```

**Main Features:**
- Split-screen interface (camera + reference signs)
- Real-time alphabet detection
- Current word and sentence display
- Speech synthesis with controls

---

## ⌨️ Keyboard Controls

| Key | Action |
|-----|--------|
| `C` | **Confirm Letter** - Add detected letter to current word |
| `S` | **Space** - Complete current word and add to sentence |
| `X` | **Clear Word** - Delete current word |
| `Z` | **Clear Sentence** - Clear entire sentence |
| `V` | **Speak** - Convert sentence to speech |
| `D` | **Download** - Save speech as `output_speech.wav` |
| `ESC` | **Exit** - Close application |

---

## 🧠 Model Architecture

```
Conv2D(32, 3x3, ReLU)
    ↓
MaxPooling2D(2x2)
    ↓
Conv2D(64, 3x3, ReLU)
    ↓
MaxPooling2D(2x2)
    ↓
Conv2D(128, 3x3, ReLU)
    ↓
MaxPooling2D(2x2)
    ↓
Flatten
    ↓
Dense(256, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(26, Softmax)
```

**Model Details:**
- **Input Shape:** (128, 128, 1) - Grayscale images
- **Output:** 26 classes (A-Z, excluding J)
- **Optimizer:** Adam
- **Loss:** Categorical Crossentropy
- **Confidence Threshold:** 50% (configurable)

---

## 🔧 Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python** | Core programming language |
| **TensorFlow/Keras** | Deep learning framework |
| **OpenCV** | Computer vision and image processing |
| **NumPy** | Numerical computations |
| **pyttsx3** | Text-to-speech conversion |
| **Matplotlib** | Data visualization (training) |
| **scikit-learn** | ML utilities |

---

## 📊 Dataset Structure

The training data is organized in the following structure:

```
final_data/train/
├── A/  (300 images)
├── B/  (300 images)
├── C/  (300 images)
├── ...
└── Z/  (300 images)
```

Each folder contains preprocessed 128x128 grayscale images of hand gestures representing that letter.

**Preprocessing Pipeline:**
1. Extract ROI from webcam frame
2. Convert to grayscale
3. Apply Gaussian blur (5x5 kernel)
4. Threshold using Otsu's method
5. Resize to 128x128
6. Normalize pixel values (0-1)

---

## 🎥 How It Works

### Step-by-Step Process:

1. **Capture Frame** - Webcam captures video frame
2. **Extract ROI** - Green bounding box defines hand region
3. **Preprocess** - Convert to grayscale → Blur → Threshold
4. **Predict** - CNN model classifies the gesture
5. **Filter** - Only predictions above confidence threshold are accepted
6. **Build Sentence** - User confirms letters to form words and sentences
7. **Synthesize Speech** - Text-to-speech engine speaks the sentence

### Image Processing Pipeline:
```
Raw Frame → Flip → ROI Extraction → Grayscale → Gaussian Blur 
→ Otsu Threshold → Resize (128x128) → Normalize → Predict
```

---

## 🎯 Confidence Threshold

The system uses a **50% confidence threshold** to filter predictions:
- **Above 50%:** Letter is displayed and can be confirmed
- **Below 50%:** "Detecting..." is shown (prevents false positives)

You can adjust this in the configuration:
```python
CONF_THRESHOLD = 0.50  # Increase for higher accuracy, decrease for easier detection
```

---

## 🔮 Future Improvements

- [ ] Add support for letters J and Z (require motion tracking)
- [ ] Implement word prediction/auto-complete
- [ ] Add support for numbers (0-9)
- [ ] Include common phrases and sentences
- [ ] Mobile app version (Android/iOS)
- [ ] Multi-language support
- [ ] Real-time gesture tracking (not just static poses)
- [ ] Cloud-based model deployment
- [ ] Gesture intensity/confidence visualization
- [ ] User authentication and history tracking

---

## 🐛 Troubleshooting

### Camera Not Working
```python
# Error: "❌ Camera not working"
# Solution: Check camera permissions and availability
cap = cv2.VideoCapture(0)  # Try changing 0 to 1, 2, etc.
```

### Low Accuracy
- Ensure good lighting conditions
- Position hand clearly within the green ROI box
- Use a plain background
- Retrain model with more diverse data

### Speech Not Working
- Check audio output device
- Verify pyttsx3 installation: `pip install pyttsx3`
- On Linux, may need: `sudo apt-get install espeak`

### TensorFlow Installation Issues
```bash
# For GPU support:
pip install tensorflow-gpu==2.10.0

# For CPU-only:
pip install tensorflow==2.10.0
```

---

## 📝 Notes

- **Letters Excluded:** J (requires motion), and sometimes I due to similarity with other gestures
- **Best Performance:** Use a plain, contrasting background (dark background for light skin, vice versa)
- **Hand Position:** Keep hand within the green ROI box and avoid moving too quickly
- **Training Time:** Approximately 10-15 minutes on CPU, 2-3 minutes on GPU

---

## 👨‍💻 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- American Sign Language (ASL) alphabet reference
- TensorFlow and Keras communities
- OpenCV documentation and tutorials
- pyttsx3 library contributors

---

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

## 🌟 Show Your Support

If you found this project helpful, please give it a ⭐️!

---

**Made with ❤️ for the deaf and hard-of-hearing community**
