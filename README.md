Official implementation of our CVPR 2026 paper: **Global-Aware Edge Prioritization for Pose Graph Initialization**

<a href="https://arxiv.org/abs/2602.21963"><img src="https://img.shields.io/badge/arXiv-2602.21963-b31b1b" alt="arXiv"></a>

Authors: [Tong Wei](https://weitong8591.github.io/), [Giorgos Tolias](https://cmp.felk.cvut.cz/~toliageo/), [Jiri Matas](https://cmp.felk.cvut.cz/~matas/), [Daniel Barath](https://scholar.google.com/citations?user=U9-D8DYAAAAJ&hl=EN) |

## Updates
2026.3.31 Initial code for tests released!

## Installation
```
pytorch-lightning=2.3.1
Python=3.10.13
Pillow=10.2.0
torchvision=0.23.0
OpenCV=4.11.0
PyTorch-Geometric
pycolmap=3.10.0
gluefactory
```
## Quick Start
1. Clone our code with the submoduldes: allrank (for training loss) and VGGT (optional tests)
```
git clone --recursive git@github.com:weitong8591/global_edge_prior.git
```
2. Download our best [checkpoint](https://cmp.felk.cvut.cz/~weitong/globaledge/best.zip).

3. Run the demo of predicting global-aware edge scores and use the selected edges/nodes for reconstruction.

  ```
  python demo.py
  ```

## Full Evaluation
In inference, we run step 1 to predict the edge ranks and construct several minimal spanning trees with the promising pairs. Step 2 is used to run COLMAP on those pairs and evaluate the accuracies of the reconstructed cameras.

Step 1. Save multiple minimal spanning trees w.r.t. the edge scores predicted by our model.
```
python test.py \
  --class_model "best.ckpt" \
  --config_file "best_hparams.yaml" \
  --dataset phototourism --ks 1 2 3 5 \
  --extra 'test' \
  --save_top \
  --cluster \
  --scenes brandenburg_gate
```

Step 2. Run COLMAP on the selected image pairs from k-MSTs. Here we load the selected pairs to COLMAP for sparse reconstruction. Relative poses on all pairs are evaluated and, AUC scores are returned at thresholds={2.5, 5, 10, 20}.
```
python run_colmap.py \
  --datasets imc2023 \
  --scenes brandenburg_gate \
  --num_threads 5 \
  --data_path <data_path> \
  --pairs <image_pair_path.txt> \
  --workspace_path run_colmap/<model_name>/<k> \
  --sp_lg \
  --trees
```

## Training

In training, we use MegaDepth training scenes and its COLMAP ground truth to optimize our model (frozen DINOv2+SALAD/MegaLoc aggregators and GNN-based global-aware edge score predictor). NDCG ranking loss is used.

Before starting, download the GT information from [here](https://cmp.felk.cvut.cz/~weitong/globaledge/megadepth_gt.zip).

```
python main.py
```

## Notes

:boom: important: before data preprocessing, create/update a 'data_dirs.yaml' file to save all the required paths.

```
dataset_dirs:
  phototourism: <src_path>
```
Configs used in COLMAP
--overwrite_reconstruction
--overwrite_database

Other configs will be explained soon.

## Acknowledgement
We appreciate the code base from [DINOv2-SALAD](https://github.com/serizba/salad.git), [MegaLoc](https://github.com/gmberton/MegaLoc.git), [COLMAP](https://github.com/colmap/colmap.git), [GLOMAP](https://github.com/colmap/glomap.git), [VGGT](https://github.com/facebookresearch/vggt.git), [glue-factory](https://github.com/cvg/glue-factory.git), [pre-commit](https://pre-commit.com/)

## Citation
More details are covered in our paper and feel free to cite it if useful:

```
@InProceedings{wei2026global,
  title={Global-Aware Edge Prioritization for Pose Graph Initialization},
  author={Wei, Tong and Tolias, Giorgos and Matas, Jiri and Barath, Daniel},
  booktitle={CVPR},
  year={2026}
}
```
Contact me at weitongln@gmail.com or weitong@fel.cvut.cz.
