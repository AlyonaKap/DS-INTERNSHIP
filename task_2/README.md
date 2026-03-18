# Task 2: Animals Detection (DistilBert + ResNet50)

## Overview

This project implements a multi-modal ML pipeline that verifies whether a given text description matches the animal depicted in an image. The pipeline integrates a ResNet50 for image classification and a DistilBert model for Named Entity Recognition, combining their outputs to produce a final boolean verification result.

## Project Structure

- `main.py`: main pipeline script that orchestrates the workflow by integrating both models, taking image and text inputs and returning a boolean match
- `ner_part/train.py`: script that loads json data from Label Studio, fine-tunes the pre-trained DistilBert model for Named Entity Recognition, and saves the learned weights and label map
- `ner_part/inference.py`: script providing the `predict_text` function that is responsible for loading the fine-tuned DistilBert model and extracting animal entities from a given text description
- `classification_part/train.py`: script that loads the animal image dataset, prepares the pre-trained ResNet50 model with a custom classification head, trains the model, and saves the model and classes
- `classification_part/inference.py`: script providing the `predict_image` function that is responsible for loading the trained ResNet50 model and predicting the animal class for a given image
- `solution.ipynb`: Jupyter notebook with exploratory data analysis, training, evaluation and edge case analysis

## Models

- **DistilBert**: a lightweight version of BERT provided by Hugging Face. It is trained on the same data as the original model but is approximately twice as small and faster, thanks to knowledge distillation: a process where a smaller model learns to replicate the behavior of a larger one. It was further pre-trained on manually collected data annotated using Label Studio.


- **ResNet50**: a 50-layer Convolutional Neural Network trained on the large-scale ImageNet dataset. In this project, we use a pre-trained version of ResNet50 and freeze its weights, allowing us to utilize its powerful feature extraction capabilities while reducing training time and computational cost. It was further pre-trained on data from 15 classes of the Animals Detection Images Dataset from Kaggle.

## Setup and Installation

*Note: This project is implemented in Python 3.12*

1. Clone the repository

```
git clone https://github.com/AlyonaKap/DS-INTERNSHIP.git
cd task_2
```

2. Create and activate a virtual environment (Recommended)

Linux 
```
python3 -m venv venv
source venv/bin/activate
```

Windows 
```
python -m venv venv
venv\Scripts\Activate.ps1
```
3. Install dependencies

```
pip install -r requirements.txt
```
4. Download Models

Download models.zip from [Download models](https://drive.google.com/file/d/1bQSDtC2N1H-CvHRVfxq_UKcxEpjNw81A/view?usp=sharing) and put into /task_2

5. Run the task

You can open and run cells in the provided Jupyter notebook:

```
jupyter notebook solution.ipynb
```

Alternatively, you can run Python script from terminal to get a prediction from ML pipeline:

Default params (image_dir = "demo\tiger.png", text = "I see a pig"):
```
python main.py 
```
Custom image and text:
```
python main.py --image_dir "path_to_image" --text "text_to_recognize"
```
## Results

| Cases | Number of correct predictions |
|-------|----------|
| True Positive | 5/5 |
| True Negative | 5/5 |
| Negative Context | 0/5 |
| Unknown Classes | 0/5 |

The pipeline excels at recognizing training classes with 100% accuracy, but evaluation of edge cases reveals specific limitations in semantic logic and number of classes. These results can be significantly improved by expanding the training dataset to include more diverse classes and fine-tuning the NLP component to recognize negative linguistic contexts




