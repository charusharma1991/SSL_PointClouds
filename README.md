# Self-Supervised Few-Shot Learning on Point Clouds

### About
This is a source code for our [NeurIPS 2020](https://nips.cc/Conferences/2020/) paper: [Self-Supervised Few-Shot Learning on Point Clouds](https://arxiv.org/abs/2009.14168). We propose two novel self-supervised pre-training tasks that encode a hierarchical partitioning of the point clouds using a cover-tree, where point cloud subsets lie within balls of varying radii at each level of the cover-tree.

### Requirements

Please download miniconda and create an environment using the following command:
```
conda create -n pytorch35
```
Activate the environment before executing the program as follows:
```
source activate pytorch35
```
### Installation
Run the following command to install the dependencies required to run the code.
```
python -r requirements.txt
```
If your default python version is 2.X, we request you to switch to 3.X.

### Usage
The directory End2End contains the end to end pipeline to run the code. 
Run the following command to reproduce the results on Celegans dataset.
```
bash end2end.sh
```
The bash file currently has code for Celegans dataset, provided as an example. You can change the dataset name and corresponding hyperparameters from the table given in the paper to reproduce the results.
This directory has the complete pipeline, from generating the spectral walks, running the par2vec model to learn primary node embeddings and finally running the link prediction model.

The src directory contains the code just for link prediction and node classification with precomputed spectral-biased random walk and paragraph vector model output.
Run the following command to run just the link prediction part of the entire pipleline
```
python LP.py
```
The LP.py is the python script which will run the link prediction model on all datasets and store the results in a text file inside the directory result_logs/LP.

The NC.py is the python script which will run the node classification model on the datasets- cora and citeseer and store the results in a text file inside the directory result_logs/NC.


### Citation
Please cite the paper if you use this code.
```
@misc{sharma2020selfsupervised,
      title={Self-Supervised Few-Shot Learning on Point Clouds}, 
      author={Charu Sharma and Manohar Kaul},
      year={2020},
      eprint={2009.14168},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
