# 🐱🐶 Cat vs Dog Deep Learning Image Classification

A deep learning-based image classification project built using **PyTorch** and **Streamlit**, capable of predicting whether an uploaded image contains a Cat or a Dog

---

## 🚀 Live Demo
You can try the live version of the app here:  
**[Cat vs Dog Classification Web App]()**  

---

## 📌 Overview

This project is a binary image classification task where a **Convolutional Neural Network (CNN)** is trained to distinguish between cats and dogs.  
The model is built using **PyTorch**, trained on a labeled dataset of cat and dog images, and deployed via **Streamlit** for real-time predictions.



---


## 🛠 Features

 - 📂 Image upload via a simple web interface.

 - 🖼 Image preprocessing (resize & normalization) before prediction.

 - 📊 Confidence score for predictions.

 - ⚡ GPU support for faster inference.


---


## 📦 Installation

### 1️⃣ Clone the repository
  ```
  git clone https://github.com/YamenRM/Cat-VS-Dog-DL-Classification.git
  cd Cat-VS-Dog-DL-Classification
  ```
### 2️⃣ Install dependencies
  ```
  pip install -r APP/requirements.txt
  ```


## 🚀 Running the Streamlit App
  ```
  streamlit run APP/app.py
  ```


---


## 📊 Model Performance
 | Class        | Precision | Recall | F1-Score |
 | ------------ | --------- | ------ | -------- |
 | Cats         | 0.70      | 0.80   | 0.75     |
 | Dogs         | 0.77      | 0.65   | 0.71     |
 | **Accuracy** | **0.73**  | -      | -        |



---



## 📊 Model Details

 - Architecture: Custom CNN with convolutional, pooling, and fully connected layers.

 - Input size: 64×64 pixels, RGB.

 - Output: 2 classes (Cat, Dog).

 - Framework: PyTorch.

 - Training: Done on custom dataset from **[Cat vs Dog dataset ]( https://www.kaggle.com/datasets/tongpython/cat-and-dog)**.


---

### 👨‍💻 Author

 - YamenRM

 - 📧 Email:yamenrafat132@gmail.com

 - Palestine | GAZA

 - 3RD YEAR AT UP

**💪 Stay Strong** 
