# MMDS_MasterThesis
:butterfly: This repository maintains the code for Master Thesis at Universität Mannheim. The thesis is conducted under the chair of Artificial Intelligence and focuses on audio processing. The aim of the thesis is to observe the interdependency between acoustic scene classification and acoustic event classification tasks. Both these tasks have been primarily observed separately but it is intuitive that one relies on the other.

## :open_file_folder: Folder contents:
  1. "audio_processing" contains python scripts for extracting panns inference from audio ".wav" files and log mel extraction script.
  2. "csv_parser" : contains jupyter notebooks of general analysis observed on scapper and tut18 dataset. The notebooks focus on exploring the distribution of events across different acoustic scenes.
  3. "models" has code setup for two types of models: Autoencoder(TAU2019_autoenc) and Acoustic Scene Classification(asc).