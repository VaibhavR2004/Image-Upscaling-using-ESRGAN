# 🖼️ ESRGAN Image Enhancement

This project enhances and upscales low-resolution images using **ESRGAN (Enhanced Super-Resolution Generative Adversarial Network)**. It applies deep learning to reconstruct high-resolution images with sharp details that traditional methods often fail to preserve.

---

## 🚀 Features

- 🔍 Super-resolves low-resolution images using a deep GAN model.
- 🧠 Utilizes a pre-trained ESRGAN model for high-quality image enhancement.
- ⚙️ Built with PyTorch and OpenCV for fast and effective processing.

---

## 🛠️ Tech Stack

- **Python** – Main programming language  
- **PyTorch** – For loading and running the ESRGAN model  
- **OpenCV** – For image I/O and processing  
- **NumPy** – For numerical operations  
- **ESRGAN** – Deep learning model for image super-resolution

---

## 📁 Project Structure
```
Image_Enhancement/
│
├── Image_Enhancement.ipynb       # Main notebook for image enhancement 
├── models/                       # Folder for ESRGAN model weights 
│   └── RRDB_ESRGAN_x4.pth        # Pre-trained ESRGAN model
├── input/                        # Folder for input low-resolution images
├── output/                       # Folder for output high-resolution images
└── README.md                     # Project documentation

```

---

## 🖥️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Image_Enhancement.git
cd Image_Enhancement
pip install -r requirements.txt

python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

pip install torch torchvision opencv-python numpy matplotlib
pip install -r requirements.txt
torch
torchvision
opencv-python
numpy
matplotlib

```
##📥 Download Pre-trained ESRGAN Model
Download the pre-trained model file from the official ESRGAN GitHub:

Download RRDB_ESRGAN_x4.pth

Place it in the models/ folder of this project:

bash
Copy
Edit
mkdir -p models
mv RRDB_ESRGAN_x4.pth models/

##🧪 Run the Notebook
Open and run the notebook in your preferred environment:

```bash
Copy
Edit
jupyter notebook Image_Enhancement.ipynb
```

##📄 License
This project is licensed under the MIT License.
MIT License
```
Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights  
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell  
copies of the Software, and to permit persons to whom the Software is  
furnished to do so, subject to the following conditions:

[...full MIT license can be added here or in a LICENSE file...]

