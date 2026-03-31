import torch
import re
import pickle
import argparse
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from pytorch_lightning.callbacks import Callback


def global_tokens2scores(descriptors):
    scores = torch.einsum('id,jd->ij', descriptors, descriptors) 
    N = descriptors.shape[0]
    diagonal = torch.eye(N, N).bool().to(scores.device)
    scores.masked_fill_(diagonal, -torch.inf)
    return scores

import sqlite3
from pathlib import Path

def load_colmap_pairs(db_path):
    
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()

    images = {}
    pairs_path = []
    cursor.execute("SELECT image_id, camera_id, name FROM images;")
    for row in cursor:
        image_id = row[0]
        image_name = row[2]
        images[image_id] = image_name

    cursor.execute(
        "SELECT pair_id, data FROM two_view_geometries WHERE rows>=?;",
        (0,),
    )
    for row in cursor:
        pair_id = row[0]
        # import pdb; pdb.set_trace()
        try:
            inlier_matches = np.fromstring(row[1], dtype=np.uint32).reshape(-1, 2).shape[0]
        except:
            inlier_matches = 0
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        image_name1 = images[image_id1]
        image_name2 = images[image_id2]
        
        pairs_path += [(image_name1, image_name2, inlier_matches) ]#, 


    return pairs_path


def convert_labels2dict(gt_labels, scene_name):

    result = {
        "pos": [],
        "neg": []
    }

    for row in gt_labels:
        img1, img2 = row[0], row[1]
        label = row[-2] 

        img1 = img1.replace(f"{scene_name}/", "")
        img2 = img2.replace(f"{scene_name}/", "")

        pair = (img1, img2)

        if label == '1':
            result["pos"].append(pair)
        else:
            result["neg"].append(pair)
    return result

def convert_labels2dict_threshold(gt_labels, scene_name):

    result = {
        "pos": [],
        "neg": []
    }

    for row in gt_labels:
        img1, img2 = row[0], row[1]
        label1 = row[-2] 
        label2 = row[-1] 

        img1 = img1.replace(f"{scene_name}/", "")
        img2 = img2.replace(f"{scene_name}/", "")

        pair = (img1, img2)

        if (label1>20) & (label2>20):
            result["pos"].append(pair)
        else:
            result["neg"].append(pair)
    return result



def load_gt_imc(scene, gt_folder='gt_tri_inliers', binary=True):
    image_list_saved = np.loadtxt(f"/mnt/personal/weitong/cache/imc2023/train/image_list/{scene}.txt", dtype=object)
    name2idx = {name: idx for idx, name in enumerate(image_list_saved)}
    gt_labels = []
    #triangulated inliers
    # GT_loaded = np.load(f"/mnt/personal/weitong/cache/imc2023/train/{gt_folder}/{scene}_inliers.npz")['inliers']
    # # ransac inliers
    # src = Path(f"/mnt/personal/weitong/cache/all/imc2023/phototourism")
    # ransac_supervision = np.loadtxt(src/f"{scene}/supervision.txt", dtype=float, usecols=[2, 3, 4])
    # ransac_pairs = np.loadtxt(src/f"{scene}/supervision.txt", dtype=object, usecols=[0, 1])
    with open(f'/mnt/personal/weitong/cache/all_pkls/{scene}.pkl', 'rb') as f:
        GT = pickle.load(f)

    for key, values in GT.items():
        img0, img1 = split_file_name(key)
        if img0 in name2idx and img1 in name2idx:
            # i, j = name2idx[img0], name2idx[img1]
            # values[0]#["ransac_inliers"]
            # values[1]
            gt_labels += [[img0, img1, values[1], values[0]]]
    if binary:
        return convert_labels2dict_threshold(gt_labels, scene)
    else:
        return gt_labels

def load_gt_imc_npz(scene, 
                    gt_folder='/mnt/personal/weitong/cache/imc2023/train/gt_tri_inliers', #'/mnt/personal/weitong/cache/megadepth_runs_all/gt', 
                    gt_folder1= '/mnt/personal/weitong/cache/imc2023/train/gt_tri_inliers',#"/mnt/personal/weitong/cache/megadepth_runs_all/gt_tri_inliers", 
                    src_path =None, 
                    all_list_path="/mnt/personal/weitong/cache/megadepth_runs_all/image_list/",
                    binary=True):
    
    #triangulated inliers
    score_matrix = np.load(f"{gt_folder}/{scene}_inliers.npz")['inliers']
    # ransac inliers
    class_score_matrix = np.load(f"{gt_folder1}/{scene}_inliers.npz")['inliers']
    subset_image_list = np.loadtxt(f"{src_path}", dtype=str).tolist() if type(src_path) is str else src_path
    all_image_list = np.loadtxt(f"{all_list_path}/{scene}.txt", dtype=str).tolist()

    all_image_list = [i.replace(f"{scene}_", '')for i in all_image_list]
    subset_image_list = [i.replace(f"{scene}_", '')for i in subset_image_list]
    subset_idx = [all_image_list.index(img) for img in subset_image_list if img in all_image_list]

    gt_dict = score_matrix[np.ix_(subset_idx, subset_idx)]
    gt_dict_class = class_score_matrix[np.ix_(subset_idx, subset_idx)]
   

    return gt_dict, gt_dict_class





def plot_image_pairs(src_root, records, figsize_per_pair=(10, 4), extra=""):

    n = len(records)
    fig, axes = plt.subplots(
        n, 2,
        figsize=(figsize_per_pair[0], n * figsize_per_pair[1])
    )

    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (scene, left, right, inliers) in enumerate(records):
        left_path = os.path.join(src_root, scene, extra, left)
        right_path = os.path.join(src_root, scene, extra, right)

        for j, (img_path, name) in enumerate([(left_path, left), (right_path, right)]):
            ax = axes[i, j]
            try:
                img = Image.open(img_path).convert("RGB")
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, "Load failed", ha="center", va="center")
            ax.axis("off")
            ax.set_title(name, fontsize=9)

        # 在 pair 中间写 inliers
        axes[i, 0].text(
            1.05, 0.5,
            f"inliers = {inliers}",
            transform=axes[i, 0].transAxes,
            va="center",
            fontsize=10,
            color="red",
            fontweight="bold"
        )

        # scene 写在最左
        axes[i, 0].set_ylabel(scene, fontsize=10)

    plt.tight_layout()
    plt.savefig("failure_pairs.png", dpi=200, bbox_inches="tight")
    plt.close()


def load_gt_mega(scene, image_list, subset_num):
    GT = load_gt_imc_npz(scene, subset_num=subset_num, gt_folder="/mnt/personal/weitong/cache/megadepth_runs_all/gt_tri_inliers/",gt_folder1="/mnt/personal/weitong/cache/megadepth_runs_all/gt_tri_inliers/")
    gt_labels = []
    for i0, img0 in enumerate(image_list):
        for i1, img1 in enumerate(image_list):
            if img0 != img1 :
                gt_labels += [[img0, img1, GT[i0][i1], GT[i0][i1]]]
    return gt_labels

class ClearCacheCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()
            print(f"Epoch {trainer.current_epoch}: CUDA cache cleared")
            
    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % 5 == 0:
            torch.cuda.empty_cache()


def find_negative_edges(scene, pairs, find_key='neg', imc=True, subset_num=0):
    our_pairs = np.loadtxt(pairs, dtype=object)#

    gt_labels_dict = load_gt_imc(scene) if imc else load_gt_lamar(scene, our_pairs, subset_num=subset_num)

    wrong = 0
    num = 0
    all = 0
    for (left, right) in our_pairs:
        if ((left, right) in  gt_labels_dict['pos']) or ((right, left) in  gt_labels_dict['pos']) or ((left, right) in  gt_labels_dict['neg'])  or ((left, right) in  gt_labels_dict['neg']):
            all += 1
    for (left, right, num) in our_pairs:
        if ((left, right) in  gt_labels_dict[find_key]) or ((right, left) in  gt_labels_dict[find_key]):#trained_tune_377_flyflyflynosaladupdatedisttop5normdynupdates_check3
            wrong += 1
    
    return wrong, all

def plot_correlation(selected_edges, GT, scores_matrix, mask, scene, class_model, oracle_edges=None, extra='', plot_extra='', fp=0):
    all =  []
    for key in selected_edges.keys():
        all += selected_edges[key]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    colors_ = ['orange', 'orange', 'orange', 'orange', 'orange']

    markers = ['o', 's', '^', 'x', 'D']
    markers_ = ['x', 'x', 'x', 'x', 'x']

    sel = selected_edges

    GT_np = GT.cpu().numpy()
    S_np = scores_matrix.detach().cpu().numpy()
    GT_rank = rankdata(-GT_np, method='ordinal').reshape(GT_np.shape)
    S_rank = rankdata(-S_np, method='ordinal').reshape(S_np.shape)

    # GT_rank = np.array([rankdata(-row, method='ordinal') for row in GT_np])
    # S_rank = np.array([rankdata(-row, method='ordinal') for row in S_np])
    N = scores_matrix.shape[0]

    for k in sel.keys():   # k in [0,1,2,3,4]
        pairs = sel[k]     # list of (i,j)oracle[k]#
        xs, ys = [], []
        plt.figure(figsize=(6,6))
        plt.scatter(GT_rank[mask],S_rank[mask],s=3, alpha=0.4, label="all pairs")

        for (i,j) in pairs:
            xs.append(GT_rank[i,j])
            ys.append(S_rank[i,j])

        plt.scatter(xs, ys,
                    s=25,
                    color=colors[k],
                    marker=markers[k],
                    label=f"selected {k+1}")
        
        if oracle_edges is not None: 
            oracle = oracle_edges
            pairs = oracle[k]  
            xs, ys = [], []

            for (i,j) in pairs:
                xs.append(GT_rank[i,j])
                ys.append(S_rank[i,j])

            plt.scatter(xs, ys,
                        s=25,
                        color=colors_[k],
                        marker=markers_[k],
                        label=f"selected oracle")
        plt.xlabel("GT")
        plt.ylabel("our ranks")
        plt.title(f"GT vs rank Corr. ({N} images, Spearman={plot_extra:.2f},FP={fp})")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        Path(f'ours_figs_oraclerank1/{scene}{extra}').mkdir(exist_ok=True, parents=True)
        plt.savefig(f'ours_figs_oraclerank1/{scene}{extra}/{class_model.replace("last.ckpt", "corr").replace("/", "_")}_{k}.png')
        plt.clf()
    



def save_mst_pairs(scores_matrix, image_list, scene, ks=[1, 2, 3, 5], save_path=None, suffix=''):

    """
        save the edges selected by minimal spanning trees (separately with removal). 
        
    """

    # initialize the graph with the given weights on edges
    G = CreateGraph(1-scores_matrix.cpu().numpy())
    return_edges = {}
    all_edges_cc = []

    # repeatedly run MST
    for k_i in range(max(ks)):   
        find_edges_, ransac_times, reject_times, _ = kruskal_optimized(G)
        find_edges = []
        for (i, j, _) in find_edges_:
            find_edges += [(i, j)]
            G.remove_edge(i, j)
        return_edges[k_i] = find_edges
        all_edges_cc += find_edges

        if k_i+1 in ks:
            with open(save_path/Path(f'{scene}_pairs_{k_i+1}_trees_{suffix}.txt'), "w") as doc:
              for _, (edge_i, edge_j) in enumerate(all_edges_cc):
                  doc.write(f"{image_list[edge_j]} {image_list[edge_i]}\n")
                  
    return return_edges



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='')
    parser.add_argument(
        "--datasets", nargs="+", default=["imc2023"]
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=['phototourism'],
        help="Categories to evaluate, if empty all categories are evaluated.",
    )
    parser.add_argument(
        "--scenes",
        nargs="+",
        default=[],
        help="Scenes to evaluate, if empty all scenes are evaluated.",
    )
    parser.add_argument("--run_path", default=Path("/mnt/personal/weitong/cache/runs_glomap_all"))#Path(__file__).parent / "runs")#
    # parser.add_argument(
    #     "--run_name",
    #     default=datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    # )
    parser.add_argument(
        "--overwrite_database", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_reconstruction", default=False, action="store_true"
    )
    parser.add_argument(
        "--overwrite_alignment", default=False, action="store_true"
    )
    parser.add_argument(
        "--sp_lg", default=False, action="store_true"
    )
    parser.add_argument(
        "--trees", default=False, action="store_true"
    )
    parser.add_argument(
        "--reestimate_by_superransac", default=True, required=False
    )
    parser.add_argument("--colmap_path", default="/mnt/appl/software/COLMAP/3.10-foss-2023b-CUDA-12.4.0/bin/colmap", required=False)
    parser.add_argument("--use_gpu", default=True, action="store_true")
    parser.add_argument("--use_cpu", dest="use_gpu", action="store_false")
    parser.add_argument("--num_threads", type=int, default=-1)
    parser.add_argument("--k", type=int, default=40)
    parser.add_argument("--quality", default="high")
    parser.add_argument("--pairs", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--database_path", default="")
    parser.add_argument("--max_num_tracks", type=int, default=-1)
    parser.add_argument("--workspace_path", default="/mnt/personal/weitong/cache/runs_glomap_all/salad_new_1_rank_nk", required=False)
    parser.add_argument("--ba_refine_extra_params", default=None)
    parser.add_argument("--glomap", default=False, action="store_true")
    parser.add_argument("--image_list_path", default=None)

    parser.add_argument(
        "--error_type",
        default="relative",
        choices=["relative", "absolute"],
        help="Whether to evaluate relative pairwise pose errors in angular "
        "distance or absolute pose errors through GT alignment.",
    )
    parser.add_argument(
        "--rel_error_thresholds",
        type=float,
        nargs="+",
        default=[2.5, 5, 10, 20],
        help="Evaluation thresholds in degrees.",
    )
    parser.add_argument(
        "--abs_error_thresholds",
        type=float,
        nargs="+",
        default=[1, 5],
        help="Evaluation thresholds in meters.",
    )
    args = parser.parse_args()
    # args.data_path = Path("/mnt/personal/weitong/cache/").resolve()
    args.colmap_path = Path(args.colmap_path).resolve()
    if args.overwrite_database:
        print("Overwriting database also overwrites reconstruction")
        args.overwrite_reconstruction = True
    if args.overwrite_reconstruction:
        print("Overwriting reconstruction also overwrites alignment")
        args.overwrite_alignment = True
    return args