# Raman- Supervised and SSL
This repository contains the code used for my analysis of Raman spectroscopy signals via Deep Learning techniques, i implemented 1-D CNNs, 1-D Transformers, 2d-CNN on 
#CWT transformed data and Hybrid variations of CNNs (for feature extractions) and Transformers for supervised classification. Moreover, i implemented such architectures for Self-Supervised 
representation learning, specifically utilising Siamese Networks for Zero-shot classification of several different spectra datasets. 
The code is structured as follows: 
#a "model" module contains different architectures import
#a "support functions" module contains functions for evertyhing from pre-processing to validation and training. 
#several instantiations of models in separate documents, where specific data extraction and preparation is made, as well as hyperarameter tuning experiments using bayesian optimization
