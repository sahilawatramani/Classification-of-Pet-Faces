###  Cat and Dog Breed Classification Using ResNet50

---

This project implements a deep learning model for classifying images of cats and dogs into different breeds. The model uses transfer learning with a pre-trained ResNet50 architecture. The objective is to accurately identify the breed of the animal in the given image.

#### **Dataset**

The dataset comprises images of various cat and dog breeds. The images are stored in the `./images` directory, and the file names follow the format `<breed_name>_<index>.jpg`. The following breeds are included:

- **Cat Breeds**:
  - Abyssinian
  - Bengal
  - Berman
  - Bombay
  - British Shorthair
  - Egyptian Mau
  
- **Dog Breeds**:
  - American Bulldog
  - American Pit Bull Terrier
  - Basset Hound
  - Beagle
  - Boxer
  - Chihuahua
  - English Cocker Spaniel
  - English Setter
  - German Shorthaired
  - Great Pyrenees

#### **Preprocessing**

1. **Image Loading**: Images are loaded using `tensorflow.keras.preprocessing.load_img`, and each image is resized to 224x224 pixels using TensorFlow's `resize_with_pad` method.

2. **Label Encoding**: The breed names are extracted from the file names, and each breed is encoded into a numerical label (e.g., Abyssinian = 0, Bengal = 1, etc.).

3. **One-Hot Encoding**: The labels are one-hot encoded to be used in the categorical classification.

4. **Data Splitting**:
   - The dataset is split into training, validation, and test sets using an 80-20 split for training and testing, and then 25% of the training data is used for validation.

#### **Model Architecture**

The model is built using the ResNet50 architecture with the following components:

- **Data Augmentation**: A sequential model is used to apply random flips and rotations to the input images.
- **Transfer Learning**:
  - A pre-trained ResNet50 model (trained on ImageNet) is used without the top layer (`include_top=False`) and with `pooling='avg'`.
  - The ResNet50 model's layers are frozen to prevent them from being trained further.
- **Prediction Layer**: A dense layer with 16 neurons and a softmax activation function is added for classifying the input image into one of the 16 classes (breeds).

#### **Model Compilation**

The model is compiled with the following configurations:

- **Optimizer**: Adam optimizer.
- **Loss Function**: Categorical cross-entropy.
- **Metrics**: Accuracy.

#### **Training**

The model is trained on the training set for 10 epochs, and the performance is evaluated on the validation set.

#### **Evaluation**

After training, the model is evaluated on the test set to determine its accuracy. Predictions are also made on the test set, and the results can be further analyzed.

#### **Usage**

To run the project, follow these steps:

1. **Install Dependencies**: Ensure you have Python installed along with the necessary libraries.
   
   ```bash
   pip install numpy pandas matplotlib tensorflow
   ```

2. **Run the Script**: Execute the script in a Python environment to train the model and evaluate its performance.

3. **Visualize the Results**: The training and validation accuracy/loss can be visualized using matplotlib to understand the model's learning curve.

#### **Conclusion**

This project demonstrates the use of transfer learning with ResNet50 for image classification tasks. The performance of the model can be improved with more data, better preprocessing, or fine-tuning the ResNet50 layers.

---

**Author**: Sahil Awatramani  

---
