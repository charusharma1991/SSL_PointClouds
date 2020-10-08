# Self-Supervised Few-Shot Learning on Point Clouds

### About
This is a source code for our [NeurIPS 2020] paper: [Self-Supervised Few-Shot Learning on Point Clouds](https://arxiv.org/abs/2009.14168). We propose two novel self-supervised pre-training tasks that encode a hierarchical partitioning of the point clouds using a cover-tree, where point cloud subsets lie within balls of varying radii at each level of the cover-tree.

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
