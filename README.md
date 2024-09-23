# MMDS MasterThesis: 
## "Deciphering the Relationship between Acoustic Scenes and Events through Explainable AI in Acoustic Scene Classification"
:technologist: This repository maintains the code for Master Thesis at Universität Mannheim. The thesis is conducted under the chair of Artificial Intelligence and focuses on audio processing. The aim of the thesis is to observe the interdependency between acoustic scenes and events while utilising the power of explainable AI (XAI) on the task of acoustic scene classification (ASC).

## :open_file_folder: Folder contents:
  1. `audio_processing`: This directory includes Python scripts for extracting PANNS inferences and log-mel spectrograms from .wav audio files.
  2. `csv_parser`: This directory contains Jupyter notebooks for general analysis based on the <span style="color:green;">scapper</span> and <span style="color:green;">tut18</span> datasets. The notebooks are designed to explore the distribution of events across various acoustic scenes.
  3. `models`: This directory contains code setups for two models: Autoencoder (TAU2019_autoenc and autoenc) and Acoustic Scene Classification (asc). Both autoencoder directories include files for the architecture and training of the models, as well as scripts for performing inference with these models. Additionally, we have included Jupyter notebooks to analyze the encoder embeddings. Furthermore, the ASC folder also features 2 major models for asc - an enhanced network ResNet based ASC architecture and a simple CNN model for performing ASC on TAU2019 dataset, along with notebooks for Grad-CAM analysis and post-Grad-CAM events analysis for both Grad-CAM and Grad-CAM++ models. The notebook `gradcam_analysis.ipynb` is dedicated to plain Grad-CAM analysis, while `gradcam_analysis-newasc.ipynb` focuses on the Grad-CAM++ model.
------------
## :pushpin: Guidelines:
1. Create a python virtual environment using conda with version Python 3.10 and install the prerequisite packages.
```
conda create -n master_thesis python=3.10
conda activate master_thesis
pip install -r requirements.txt
```
