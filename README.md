# ✍️ MNIST Handwritten Digit Recognition using CNN

This project implements a Deep Learning model to accurately classify grayscale images of handwritten digits (0-9) from the ubiquitous **MNIST dataset**.

---

## 💡 Project Overview

The goal of this project is to build a robust image classification system leveraging a **Convolutional Neural Network (CNN)**. The model is trained on a massive dataset of $28 \times 28$ pixel handwritten digits to learn complex visual features and predict the correct digit class.

## 💾 Dataset

| Attribute | Details |
| :--- | :--- |
| **Name** | MNIST (Modified National Institute of Standards and Technology) |
| **Size** | 60,000 images for training, 10,000 images for testing. |
| **Format** | $28 \times 28$ grayscale images. |
| **Classes** | 10 (digits 0 through 9). |

## 🚀 Key Technologies & Architecture

* **Frameworks:** TensorFlow, Keras
* **Core Model:** Convolutional Neural Network (CNN)
* **Key Layers:**
    * **Conv2D and MaxPooling:** Used to extract spatial features and reduce dimensionality.
    * **Flatten:** Converts 2D feature maps into a 1D vector.
    * **Dense Layer with Softmax:** The output layer for final classification probabilities.

## ⚙️ How to Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [Your Repository URL]
    cd [your-repo-name]
    ```

2.  **Install Dependencies:**
    *(Ensure you have Python installed. The required libraries are listed in `requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Training Script:**
    *(This script will automatically download the MNIST dataset, train the CNN model, and save the weights.)*
    ```bash
    python train.py
    ```

4.  **Evaluate and Predict:**
    *(Check the script output to see the model's performance on the test set.)*
    ```bash
    python evaluate.py
    ```

## 📊 Results

The trained CNN model consistently achieves a classification accuracy of **[Insert Your Achieved Accuracy]%** on the MNIST test dataset.

## 🔮 Future Scope

* **GUI Integration:** Develop a user interface (using Tkinter or Streamlit) for drawing digits and getting real-time predictions.
* **Data Augmentation:** Implement techniques like rotation and scaling to make the model more robust to variations in handwriting.
* **Model Optimization:** Explore hyperparameter tuning or more complex architectures (e.g., deeper CNNs) to push accuracy higher.
