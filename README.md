# Exploring Generative Artificial Intelligence and Data Augmentation Techniques for Spectroscopy Analysis

# Installation

The software dependencies for this project are open-source and managed using [Miniconda](https://docs.anaconda.com/miniconda/) v24.1.2.
An environment YAML file is provided to create a custom environment and import the required dependencies.

Dependencies:
* Python v3.11.8
* Scikit-learn v1.5.2
* PyTorch v2.2.1
* NumPy v1.24.3
* Pandas v2.2.1
* Jupyter notebook v7.0.8

The dependencies can be installed manually or using the environment.yml file provided. See the [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf) for a quick guide on how to create a virtual environment.
Alternatively, [Google Colab](https://colab.google/) provides a free interactive Jupyter notebook environment with GPU access.

# Directories
This repository contains three folders located in the *src* directory that provide examples of methods described in our article. 
Each folder includes Jupyter notebooks that load the data from a CSV file named *cyclohexane.csv* in the *data* folder, located in the same directory as *src*.

The *Preprocessing* folder contains examples of pretreatment and preprocessing techniques corresponding to Section 3. Data Preprocessing.

The *Augmentation* folder contains examples of statistical and data augmentation methods, as described in Section 4. Chemometrics and Data Augmentation.

The *Generative AI* folder contains code that demonstrates how to train a Generative Adversarial Network and a Variational Autoencoder. 
The VAE demonstrates how to apply raw versus preprocessed data, and the GAN demonstrates dense versus convolutional model architectures.

# Contact Us
For any questions related to this work, please contact the authors:
* A.flanagan18@universityofgalway.ie
* frank.glavin@universityofgalway.ie

# Acknowledgements
* This work was supported by the Taighde Éireann - Research Ireland Centre for Research Training in Artificial Intelligence (Grant No. 18/CRT/6223)
* This work was supported by the University of Galway
* The authors wish to express their sincere gratitude to Dr. Matthew Webberley and SK Pharmteco for providing the necessary facilities and supporting the data acquisition
