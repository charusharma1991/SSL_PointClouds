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
### Usage
This code has three parts to run. 
1. Generate covertree and ball pair data for self-supervised labels. Refer directory "Covertree".
2. Self-supervised learning in a FSL set up. Refer directory "SSL".
3. Point cloud classification using learned point cloud embeddings in step 2. Refer directory "Classification".

Run the following commands to reproduce the results on Sydney dataset.
1. First, go to the directory "Covertree" and run:
```
python test_covertree.py 2.0
```
Here "2.0" is the base of the radius for covertree. This will generate data for SSL in dataset directory under "dict". We use the base code for covertree generation from [here](https://github.com/n8epi/CoverTree).

2. Go to the "SSL" directory and run:
```
python train_FSL.py
```
This will generate model using our two self-supervised learning tasks.

3. Finally, go to the directory "Classification" and run:
```
python train_classifier.py
```
There are DGCNN and PointNet networks in this directory for downstream classification task.
We thank the authors of DGCNN[1] and PointNet[2] for providing their code.

[1] Wang, Yue, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, and Justin M. Solomon. "Dynamic graph cnn for learning on point clouds." Acm Transactions On Graphics (tog) 38, no. 5 (2019): 1-12.

[2] Qi, Charles R., Hao Su, Kaichun Mo, and Leonidas J. Guibas. "Pointnet: Deep learning on point sets for 3d classification and segmentation." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 652-660. 2017.

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
