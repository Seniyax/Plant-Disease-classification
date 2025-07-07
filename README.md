# Plant Disease classification
This Project implements a deep-learning based Plant disease classification system using PlantVillage dataset.It leverages a fine-tuned **EfficientNet-B0** model to classify plant diseases across 38 classes, achieving 96% accuracy on the validation set. The system includes a user-friendly Streamlit web interface for image uploads and predictions, and a Flask API for programmatic access. The project is developed in Python using PyTorch, with a focus on modularity and ease of use.
## Project Overview
The goal is to classfy plant diseases from leaf images to aid farmers and researchers in early detection and management. The project is divided into three stages:
 
 1. **Data Preprocessing and Understanding** - Load and Preprocessed the PlantVillage data with data agumentation and normalization.
 2. **Model Fine-Tunning and Evaluation** - Fine-tuned a pre-trained EfficientNet-B0 model, achieving 96% accuracy, with detailed metrics (precision, recall, F1-score) and a confusion matrix.
 3. **UI and API Development** - Built a Streamlit UI for interactive predictions and a Flask API for programmatic access, with integration between the two.
## Dataset
- **Source** - PlantVillage [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- **Size** - ~54,000 RGB images (256x256 pixels) across 38 classes
- **Structure** - Organized in folders by class under 'PlantVillage dataset/color/'
- **Preprocessing**:
  - Training:Resized to 224x224,applied augmentation (random flips,rotations) and normalized imagenet stats
  - Validataion:Resized to 224x224,normalized without augmentation
  - Split:80% trainning and 20% validation
## Model
- **Architecture** - EfficientNet-B0 (pretrained on Imagenet)
- **Fine-Tunning** :
   - Modified the classifier to output 38 classes.
   - Intially Trained the classifier only,then fine-tuned the last few layers.
   - Used Adam Optimizer (learning rate,1e-3 for classifier,1e-4 for fine-tunning) with Cross Entrophy Loss
- **Performance** :
   - Validation Accuracy 96%
   - F1 score 0.95
     
     ![Network-structure-of-the-EfficientNet-B0-model](https://github.com/user-attachments/assets/799ab36b-0b5b-4be3-b11d-66dcb62ef080)

 ## Project Structure
 ```bash
 plant_disease_classifier/
├── venv/                           # Virtual environment
├── app.py                         # Streamlit UI script
├── api.py                         # Flask API script
├── requirements.txt               # Python dependencies
├── plant_disease_model.ipynb     # notebook includes the training and evaluvation
│── plant_disease_model.pth    # Fine-tuned EfficientNet-B0 weights
│── plantvillage_class_mapping.json  # Class-to-index mapping
└── .gitignore                     # Git ignore file
```
## Installation
1. **Clone the Repository**
   ```bash
   git clone https://github.com/Seniyax/plant-Disease-Classification.git
   cd plant-Disease-Classification
   ````
2. **Setup a Virtual Enviroment or Conda**
   ```bash
   python -m venv venv
   conda create my_env python = 3.7 # for conda
   ````
3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ````
4. **Place the Model and Class json file**
   - Place the model.pth and class.json in the relevant directory

## Usage
**Running the Streamlit UI**
```bash
streamlit run app.py
```
**Running the Flask Backend**
```bash
python api.py
````
## Testing 
- Used new Images from internet
- Expected Accuracy 96% from validation

**Model was trained using Goggle colab's T4 GPU but for Local Inference CPU is enough**

## Futher Improvements
- Addressing Class-imbalance
- Fine-tunning more Layers
- Ensamble Learner (EffiecientNet + ResNet)


  

https://github.com/user-attachments/assets/25d2d344-d80b-48e6-a7a3-cc4168234275



  

 
