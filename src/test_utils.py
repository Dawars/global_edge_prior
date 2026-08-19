import torch
import os
import yaml
import pymetis
import argparse
import numpy as np
from models.class_model_new import EdgeRankModel
from collections import defaultdict, deque
from src.test_utils import *
from src.kruskals import *
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_class_model(ckpt_path, config_path, device='cuda'):
    with open(config_path, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    if 'num_classes' not in data_loaded['edge_config'].keys() and data_loaded['edge_arch'] =='Custom_GNN' : data_loaded['edge_config']['num_classes'] = 1
    if 'not_tune' not in data_loaded.keys(): data_loaded['not_tune'] = False

    remove_keys = ['edge_classifier.edge_filtered', 'edge_classifier.dust_bin', 'edge_classifier.filter_threshold', 'edge_classifier.sigmoid_shift', 'edge_classifier.return_tokens' ]
    model = EdgeRankModel(
        edge_arch=data_loaded['edge_arch'],
        edge_config=data_loaded['edge_config'],
        test=True
       )
    ckpt = torch.load(ckpt_path, map_location=torch.device(device))
    state_dict = ckpt["state_dict"]
    state_dict = {k: v for k, v in state_dict.items()
                  if (not k.startswith('model_old')) & (k not in remove_keys)}
    model.load_state_dict(state_dict)
    model = model.eval()
    model = model.to(device)
    print(f"Loaded model from {ckpt_path} Successfully!")
    return model


def global_tokens2scores(descriptors):
    """Computer the similarity matrix on global tokens."""
    scores = torch.einsum('id,jd->ij', descriptors, descriptors)
    N = descriptors.shape[0]
    diagonal = torch.eye(N, N).bool().to(scores.device)
    scores.masked_fill_(diagonal, -torch.inf)
    return scores


def adjacency_matrix_to_graph(adj_matrix, threshold=0):
    """Converts an adjacency (score) matrix into a graph representation for METIS.

    Removes weak connections if a threshold is set.
    """
    n = adj_matrix.shape[0]
    adj_matrix = np.maximum(adj_matrix, adj_matrix.T)

    adjacency_list = []

    for i in range(n):
        neighbors = [j for j in range(n) if adj_matrix[i, j] > threshold and i != j]
        adjacency_list.append(neighbors)

    return adjacency_list

def partition_graph(adj_matrix, k=2, threshold=0):
    """Partitions the graph using METIS into k clusters."""
    adjacency_list = adjacency_matrix_to_graph(adj_matrix, threshold)
    n_cuts, partitions = pymetis.part_graph(k, adjacency=adjacency_list)

    return partitions


def precision_at_k(y_true, y_pred, k):
    """Calculate precision at k for a single sample.

    Args:
        y_true (list): Ground truth (true relevance labels)
        y_pred (list): Predicted scores or rankings
        k (int): Number of top results to consider

    Returns:
        float: Precision@k
    """
    # Sort predictions and get top k indices
    top_k_indices = sorted(range(len(y_pred)), key=lambda i: y_pred[i], reverse=True)[:k]

    # Count relevant items in top k
    relevant_in_top_k = sum(1 for i in top_k_indices if y_true[i] == 1)

    # Calculate precision
    return relevant_in_top_k / k



def graph_clustering(scores, k_init=3, k_max=10, max_edges=400*399/2):
    k_init = 3
    while True:
        partitions = np.array(partition_graph(scores, k_init))
        clusters = set(partitions)
        indices = []
        for cluster in clusters:
            indices += [(partitions==cluster).nonzero()[0].flatten()]
        # import pdb; pdb.set_trace()
        if max([index.shape[0] for index in indices]) > 400:
            k_init += 1
        elif k_init >= k_max:
            break
        else:
            break

    return partitions, clusters



def graph_clustering_nn(pretrained_salad_scores, num_clusters, k_neighbors=1):
    partitions = np.array(partition_graph(pretrained_salad_scores, num_clusters))
    clusters = set(partitions)
    indices = []

    for cluster in clusters:
        idx = np.where(partitions == cluster)[0].tolist()
        expanded = set(idx)
        nn = torch.topk(pretrained_salad_scores[idx], k_neighbors, dim=1).indices.T.flatten().tolist()
        expanded.update(nn)
        if k_neighbors ==1:
            indices.append(np.array(list(expanded)))
        else:
            indices.append(np.array(list(expanded)[:500]))

    return indices

def graph_cluster_connect(pretrained_salad_scores, num_clusters, k_neighbors=1, limit=500):
    indices = []
    while True:
        indices = graph_clustering_nn(pretrained_salad_scores, num_clusters, k_neighbors)
        if max(len(idx) for idx in indices) > limit:
            num_clusters += 1
        else:
            break
    return indices


def save_knn_pairs(scores_matrix, ks=[1, 2, 3, 5, 10], save_path=None, image_list=None, remove_prefix='', suffix=''):
    for k in ks:
        N = len(image_list)
        topk = torch.topk(scores_matrix, k) if N >= k else torch.topk(scores_matrix, N)
        with open(save_path/Path(f'ours_pairs_{k}_matrix_{suffix}.txt'), "w") as doc:
            for i, name in enumerate(image_list):
                for j in topk.indices[i]:
                    doc.write(f"{name.replace(f'{remove_prefix}/', '')} {image_list[j].replace(f'{remove_prefix}/', '')}\n")



def build_graph(edges):
    adj = defaultdict(set)
    for e in edges:
        if isinstance(e, str):
            u, v = e.strip().split()
        else:
            u, v = e
        adj[u].add(v)
        adj[v].add(u)
    return adj

def shortest_path_matrix(edges):

    adj = build_graph(edges)
    nodes = sorted(adj.keys())
    n = len(nodes)
    node2idx = {node: i for i, node in enumerate(nodes)}
    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)

    # BFS for each node
    for src in nodes:
        src_idx = node2idx[src]
        queue = deque([(src, 0)])
        visited = {src}
        while queue:
            u, d = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    v_idx = node2idx[v]
                    dist[src_idx, v_idx] = d + 1
                    dist[v_idx, src_idx] = d + 1
                    queue.append((v, d + 1))
    return dist, nodes


def shortest_path_matrix_total(edges, total_nodes):
    # build adjacency
    adj = build_graph(edges)

    # ensure all nodes exist, even isolated ones
    for i in range(total_nodes):
        if i not in adj:
            adj[i] = []

    nodes = sorted(adj.keys())
    n = len(nodes)
    node2idx = {node: i for i, node in enumerate(nodes)}

    dist = np.full((n, n), np.inf)
    np.fill_diagonal(dist, 0)
    # BFS for each node
    for src in nodes:
        src_idx = node2idx[src]
        queue = deque([(src, 0)])
        visited = {src}

        while queue:
            u, d = queue.popleft()
            for v in adj[u]:
                if v not in visited:
                    visited.add(v)
                    v_idx = node2idx[v]
                    dist[src_idx, v_idx] = d + 1
                    dist[v_idx, src_idx] = d + 1
                    queue.append((v, d + 1))

    return dist, nodes


def update_dist(find_edges, total_nodes=None):
    if total_nodes is None:
        matrix = shortest_path_matrix(find_edges)
    else:
        matrix = shortest_path_matrix_total(find_edges, total_nodes)

    return matrix

def score2cost(scores_matrix):
    cost_matrix = -scores_matrix
    N = len(scores_matrix)
    diagonal = torch.eye(N, N).bool().to(cost_matrix.device)
    cost_matrix.masked_fill_(diagonal, torch.inf)

    return cost_matrix

def save_mst_pairs_update_dists(
    scores_matrix,
    ks=[1, 2, 3, 5],
    save_path=None,
    image_list=None,
    inlier_matrix=None,
    scene='',
    suffix='',
    norm=True,
    update=True,
    knn=5,
    dyn=True,
    update_thr=0.9,
):
    """Build multiple spanning trees iteratively from a score matrix and optionally update the score matrix after
    each tree."""

    N = len(scores_matrix)

    # the original score matrix
    scores_matrix_ori = scores_matrix.clone()
    scores_matrix_update = scores_matrix.clone() #the matrix that will be updated iteratively

    # Ignore self-connections
    diagonal = torch.eye(N, N, dtype=torch.bool, device=scores_matrix_ori.device)
    scores_matrix_ori.masked_fill_(diagonal, -torch.inf)
    scores_matrix_update.masked_fill_(diagonal, -torch.inf)

    all_edges_cc = []
    return_edges = {}

    # dist_matrix: save the lengths of the shortest path between every two nodes.
    dist_matrix = torch.zeros_like(scores_matrix)

    for k_i in tqdm(range(max(ks))):
        cost_matrix = -scores_matrix_update
        G = CreateGraph(cost_matrix)

        # Remove edges that have already been selected in previous spanning trees (k>0)
        if k_i > 0:
            for (i, j) in all_edges_cc:
                G.remove_edge(i, j)

        try:
            if inlier_matrix is not None:
                find_edges_, ransac_times, reject_times, _ = kruskal_optimized(G, ransac_inliers=inlier_matrix)
            else:
                find_edges_ = kruskal_optimized(G)
            find_edges = []
            weights = []
            for (i, j, w) in find_edges_:
                find_edges.append((i, j))
                weights.append(w.item())

            all_edges_cc+=find_edges
            return_edges[k_i] = find_edges

            # Update graph distances based on all selected edges so far
            dist_matrix = update_dist(all_edges_cc)[0]

            if norm:
                dist_matrix = dist_matrix / dist_matrix.max() + 1

            find_edges_idx = torch.from_numpy(np.vstack(find_edges))

            # Select candidate edges for score updating
            # top-k images w.r.t. the updated/original score matrix
            if dyn:
                # Dynamically suppress edges already selected in the current update matrix
                scores_matrix_update[find_edges_idx[:, 0], find_edges_idx[:, 1]] = -torch.inf
                col_idx = torch.topk(scores_matrix_update, knn).indices
            else:
                col_idx = torch.topk(scores_matrix_ori, knn).indices

            row_idx = torch.arange(N, device=col_idx.device).unsqueeze(1).expand(-1, knn)
            col_indices_all = torch.stack([row_idx, col_idx], dim=2).view(-1, 2)

            # Only update high-confidence edges
            col_values = scores_matrix_ori[col_indices_all[:, 0], col_indices_all[:, 1]]
            mask = col_indices_all[col_values > update_thr]

            dist_tensor = torch.from_numpy(dist_matrix).to(scores_matrix.device, scores_matrix.dtype)

            if update:
                scores_matrix_update[mask[:, 0], mask[:, 1]] = (
                    scores_matrix_update * dist_tensor
                )[mask[:, 0], mask[:, 1]]
            else:
                scores_matrix_update[mask[:, 0], mask[:, 1]] = (
                    scores_matrix_ori * dist_tensor
                )[mask[:, 0], mask[:, 1]]

        except Exception as e:
            print(e, f"no more trees, in total {k_i + 1} spanning trees!")
            break

        if k_i + 1 in ks:
            with open(save_path / Path(f"ours_{scene}_pairs_{k_i+1}_trees_{suffix}.txt"), "w") as doc:
                for edge_i, edge_j in all_edges_cc:
                    doc.write(f"{image_list[edge_j]} {image_list[edge_i]}\n")

    print(f"--------------We built {k_i + 1} spanning trees in total!----------------")
    return return_edges

def compute_requried_k(dist_max1, dist_max2, threshold=0.10):
    # old, new
    if abs(dist_max1-dist_max2)/dist_max1 < threshold:
        return False
    else:
        return True

def list2_idx(pairs, image_list):
    return [(image_list.index(pair_i), image_list.index(pair_j)) for pair_i, pair_j in pairs]

def save_mst_pairs_update_dists_numpy(
    scores_matrix,
    ks=(1, 2, 3, 5),
    save_path=None,
    image_list=None,
    inlier_matrix=None,
    suffix='',
    norm=True,
    update=True,
    knn=5,
    dyn=True,
    update_thr=0.9,
):
    """Build multiple spanning trees iteratively from a score matrix and optionally update the score matrix after
    each tree.

    Pure NumPy version for matrix ops. Graph/MST helpers (CreateGraph, kruskal_optimized, update_dist) are reused.
    """

    # ---------- input to numpy ----------
    if hasattr(scores_matrix, "detach"):  # torch tensor
        scores_matrix = scores_matrix.detach().cpu().numpy()
    else:
        scores_matrix = np.asarray(scores_matrix)

    if inlier_matrix is not None and hasattr(inlier_matrix, "detach"):
        inlier_matrix = inlier_matrix.detach().cpu().numpy()
    elif inlier_matrix is not None:
        inlier_matrix = np.asarray(inlier_matrix)

    scores_matrix_ori = scores_matrix.copy()
    scores_matrix_update = scores_matrix.copy()

    N = scores_matrix.shape[0]

    # Ignore self-connections
    np.fill_diagonal(scores_matrix_ori, -np.inf)
    np.fill_diagonal(scores_matrix_update, -np.inf)

    all_edges_cc = []
    return_edges = {}

    # dist_matrix: shortest path lengths between every two nodes
    dist_matrix = np.zeros_like(scores_matrix, dtype=np.float32)

    max_k = max(ks)

    for k_i in tqdm(range(max_k)):
        cost_matrix = -scores_matrix_update
        G = CreateGraph(cost_matrix)

        # Remove edges that have already been selected in previous spanning trees
        if k_i > 0:
            for (i, j) in all_edges_cc:
                G.remove_edge(i, j)

        try:
            if inlier_matrix is not None:
                find_edges_, ransac_times, reject_times, _ = kruskal_optimized(
                    G, ransac_inliers=inlier_matrix
                )
            else:
                find_edges_ = kruskal_optimized(G)

            find_edges = []
            for item in find_edges_:
                # compatible with (i, j, w)
                i, j = int(item[0]), int(item[1])
                find_edges.append((i, j))

            all_edges_cc.extend(find_edges)
            return_edges[k_i] = find_edges

            # Update graph distances based on all selected edges so far
            dist_matrix = update_dist(all_edges_cc)[0]

            if norm:
                max_dist = dist_matrix.max()
                if max_dist > 0:
                    dist_matrix = dist_matrix / max_dist + 1.0
                else:
                    dist_matrix = dist_matrix + 1.0

            # ---------- candidate edges for updating ----------
            if dyn:
                # suppress selected edges in current update matrix
                if len(find_edges) > 0:
                    find_edges_idx = np.asarray(find_edges, dtype=np.int64)
                    scores_matrix_update[
                        find_edges_idx[:, 0], find_edges_idx[:, 1]
                    ] = -np.inf

                # top-k per row from updated matrix
                # argpartition is faster than full argsort
                col_idx = np.argpartition(scores_matrix_update, -knn, axis=1)[:, -knn:]
            else:
                col_idx = np.argpartition(scores_matrix_ori, -knn, axis=1)[:, -knn:]

            row_idx = np.arange(N)[:, None]
            row_idx = np.broadcast_to(row_idx, col_idx.shape)

            rows = row_idx.reshape(-1)
            cols = col_idx.reshape(-1)

            # Only update high-confidence edges
            col_values = scores_matrix_ori[rows, cols]
            valid = col_values > update_thr

            rows = rows[valid]
            cols = cols[valid]

            if rows.size > 0:
                if update:
                    scores_matrix_update[rows, cols] *= dist_matrix[rows, cols]
                else:
                    scores_matrix_update[rows, cols] = (
                        scores_matrix_ori[rows, cols] * dist_matrix[rows, cols]
                    )

        except Exception as e:
            print(e, f"no more trees, in total {k_i + 1} spanning trees!")
            break

        if (k_i + 1) in ks and save_path is not None:
            out_path = Path(save_path) / Path(
                f"ours_pairs_{k_i+1}_trees_{suffix}.txt"
            )
            with open(out_path, "w") as doc:
                for edge_i, edge_j in all_edges_cc:
                    doc.write(f"{image_list[edge_j]} {image_list[edge_i]}\n")

    print(f"--------------We built {k_i + 1} spanning trees in total!----------------")
    return return_edges

def save_mst_pairs(
        scores_matrix,
        ks=[1, 2, 3, 5],
        save_path=None,
        image_list=None,
        scene='',
        suffix='',
    ):

    # Note here the input score matrix is not a cost
    N = len(scores_matrix)
    all_edges_cc = []
    return_edges = dict()
    # cost for minimal spanning tree construction
    cost_matrix = -scores_matrix
    for k_i in tqdm(range(max(ks))):

        G = CreateGraph(cost_matrix)
        if k_i > 0:
            for (i, j) in all_edges_cc:
                G.remove_edge(i, j)
        try:
            find_edges_, _, _, _ = kruskal_optimized(G)
            find_edges = []
            weights = []
            for (i, j, w) in find_edges_:
                find_edges += [(i, j)]
                weights += [w.item()]
            all_edges_cc += find_edges
            return_edges[k_i] = find_edges

        except Exception as e:
            print(e, f"no more trees, in total {k_i+1} spanning trees!")
            break

        if k_i+1 in ks:
            with open(save_path/Path(f'ours_{scene}_pairs_{k_i+1}_trees_{suffix}.txt'), "w") as doc:
              for _, (edge_i, edge_j) in enumerate(all_edges_cc):
                  doc.write(f"{image_list[edge_j]} {image_list[edge_i]}\n")

    print(f"--------------------We built {k_i+1} spanning trees in total!------------------------------")
    return return_edges


def get_edge_probs(output, edge_config):

    if 'num_classes' in edge_config:
        if edge_config['num_classes'] == 1:
            return torch.sigmoid(output.squeeze())
        else:
            return F.softmax(output.squeeze(), dim=1)[:, 1]

    return torch.sigmoid(output.squeeze())


def run_edge_classifier(tokens, class_model, args):
    aggre_weights = (
        class_model.initialize_agg_weights_tokens(tokens)
        if 'knn' in class_model.edge_config and class_model.edge_config['knn'] > 0
        else None
    )

    model_results = class_model.edge_classifier(tokens, aggre_weights=aggre_weights)

    output = model_results['edges']
    probs = get_edge_probs(output, class_model.edge_config)
    filtered_indices = model_results.get('edge_indices', None)

    return {
        'probs': probs,
        'filtered_indices': filtered_indices,
        'model_results': model_results,
    }


def infer_edges_clustered(
    tokens,
    class_model,
    args,
    out_dir,
    scene,
    cluster_size,
):
    """Graph clustering is applied when N > 500 due tp CUDA memory constraint.

    average the scores if multiple scores gieven by different subgroups.
    """
    N = len(tokens)

    num_clusters = max(round(N / cluster_size), 3)
    cluster_file = out_dir / f"{scene}_{cluster_size}clusters.npz"

    if cluster_file.exists():
        indices = np.load(cluster_file, allow_pickle=True)['indices']
    else:
        indices = graph_cluster_connect(
            global_tokens2scores(tokens).cpu(),
            num_clusters,
            k_neighbors=1,
        )
        np.savez(cluster_file, indices=np.array(indices, dtype=object))

    scores_sum = torch.zeros((N, N), device="cpu")
    count_matrix = torch.zeros((N, N), device="cpu")

    with torch.no_grad():
        for index in indices:
            batch_tokens = tokens[index]

            result = run_edge_classifier(batch_tokens, class_model, args)
            probs = result['probs']
            filtered_indices = result['filtered_indices']

            del batch_tokens
            del result['model_results']

            if filtered_indices is None:
                raise ValueError("Cluster mode requires 'edge_indices' in model_results.")

            local_pairs = index[filtered_indices.cpu()].T

            for i, pair in enumerate(local_pairs):
                u, v = pair[0], pair[1]
                score = probs[i].item() if probs.ndim > 0 else probs.item()

                scores_sum[u, v] += score
                scores_sum[v, u] += score
                count_matrix[u, v] += 1
                count_matrix[v, u] += 1

    scores_final = torch.zeros((N, N), device="cpu")
    nonzero_mask = count_matrix > 0
    scores_final[nonzero_mask] = scores_sum[nonzero_mask] / count_matrix[nonzero_mask]

    return {'probs': scores_final.to(tokens.device)}


def infer_edges_direct(tokens, class_model, args):
    with torch.no_grad():
        result = run_edge_classifier(tokens, class_model, args)

    return {
        'probs': result['probs'],
        'filtered_indices': result['filtered_indices'],
    }


def infer_edges(tokens, class_model, args, out_dir, scene):
    N = len(tokens)

    if args.cluster and N > args.max_size:
        return infer_edges_clustered(tokens, class_model, args, out_dir, scene, args.max_size)

    return infer_edges_direct(tokens, class_model, args)

def find_unreliable_nodes(
    scores_matrix,
    topk=30,
    mean_thr=0.2,
):
    """Find unreliable nodes based on how often they appear in top-k neighbors.

    1. For each row, select top-k highest scores.
    2. Count how often each node appears.
    3. Nodes that never appear are considered unreliable.
    4. Second check: if their average score (per row) is high, keep them.
    """
    S = scores_matrix.clone()

    # Top-k neighbors per row
    _, idx = torch.topk(S, k=min(topk, S.shape[1]), dim=1, largest=True)

    # Count occurrences
    best_counts = torch.bincount(idx.flatten(), minlength=S.shape[0]).float()

    # Candidates: nodes never selected
    filtered_images = (best_counts < 1).nonzero(as_tuple=True)[0]

    # Second check: remove only if mean score is low
    if filtered_images.numel() > 0:
        keep = (scores_matrix[filtered_images].mean(-1) < mean_thr).nonzero(as_tuple=True)[0]
        filtered_images = filtered_images[keep]

    mask = torch.ones(S.shape[0], dtype=torch.bool, device=scores_matrix.device)
    if filtered_images.numel() > 0:
        mask[filtered_images.to(mask.device)] = False

    return filtered_images, mask


def build_cost_matrix(scores_matrix):
    """Convert score matrix to cost matrix for MST.

    cost = 1 - score
    Diagonal is set to +inf to avoid self-connections.
    """
    dist_matrix = 1 - scores_matrix.clone()

    N = dist_matrix.shape[0]
    diagonal = torch.eye(N, dtype=torch.bool, device=dist_matrix.device)
    dist_matrix.masked_fill_(diagonal, torch.inf)

    return dist_matrix

def save_filtered_image_list(out_dir, scene, image_list):
    save_dir = out_dir / "filtered_image_lists"
    save_dir.mkdir(exist_ok=True, parents=True)

    with open(save_dir / f"{scene}.txt", "w") as f:
        f.write("\n".join(image_list))

def build_output_dir(args):
    parts = [
        f"{args.knn}nn",
        f"{args.update_thr}_{args.extra}",
        "nodeout" if args.filter_nodes else None,
    ]
    extra = "_".join([p for p in parts if p])

    out_path = Path(args.class_model).parent
    out_dir = out_path / f"image_pairs_{extra}"
    out_dir.mkdir(exist_ok=True, parents=True)
    return out_dir

def convert_world_to_cam_to_cam_to_world(extrinsics):
    """Invert world-to-camera extrinsics to obtain camera-to-world transforms."""
    T_w2c = extrinsics_to_matrix(extrinsics)
    return invert_se3(T_w2c)

def extrinsics_to_matrix(extrinsics):
    """Convert [N, 3, 4] extrinsics to homogeneous [N, 4, 4] matrices."""
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (3, 4):
        raise ValueError(f"Expected extrinsics of shape [N,3,4], got {tuple(extrinsics.shape)}")
    n = extrinsics.shape[0]
    device = extrinsics.device
    dtype = extrinsics.dtype
    mats = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).expand(n, 4, 4).clone()
    mats[:, :3, :3] = extrinsics[:, :3, :3]
    mats[:, :3, 3] = extrinsics[:, :3, 3]
    return mats

def invert_se3(T):
    """Invert batched SE3 matrices."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    Rt = R.transpose(-1, -2)
    t_inv = -(Rt @ t.unsqueeze(-1)).squeeze(-1)
    Tin = torch.eye(4, device=T.device, dtype=T.dtype).expand(T.shape)
    Tin = Tin.clone()
    Tin[..., :3, :3] = Rt
    Tin[..., :3, 3] = t_inv
    return Tin

def compute_rel_errors_from_poses(Twc_gt_dict, Twc_pred_dict, pairs):
    dRs = []
    dts = []
    valid_pairs = []

    for name_i, name_j in pairs:
        if name_i not in Twc_gt_dict or name_j not in Twc_gt_dict:
            continue
        if name_i not in Twc_pred_dict or name_j not in Twc_pred_dict:
            continue

        Twc_gt_i = Twc_gt_dict[name_i]
        Twc_gt_j = Twc_gt_dict[name_j]
        Twc_pr_i = Twc_pred_dict[name_i]
        Twc_pr_j = Twc_pred_dict[name_j]

        R_gt, t_gt = relative_pose_from_Twc(Twc_gt_i, Twc_gt_j)
        R_pr, t_pr = relative_pose_from_Twc(Twc_pr_i, Twc_pr_j)

        dR = rotation_error_deg(R_pr, R_gt)
        dt = translation_error_deg(t_pr, t_gt)

        dRs.append(dR)
        dts.append(dt)
        valid_pairs.append((name_i, name_j))

    return np.array(dts), np.array(dRs), valid_pairs


def relative_pose_from_Twc(Twc_i, Twc_j):
    T_ij = np.linalg.inv(Twc_j) @ Twc_i
    R_ij = T_ij[:3, :3]
    t_ij = T_ij[:3, 3]
    return R_ij, t_ij

def rotation_error_deg(R_pred, R_gt):
    R_err = R_pred @ R_gt.T
    cos = (np.trace(R_err) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))

def translation_error_deg(t_pred, t_gt, eps=1e-8):
    t_pred = np.asarray(t_pred)
    t_gt = np.asarray(t_gt)

    n1 = np.linalg.norm(t_pred)
    n2 = np.linalg.norm(t_gt)
    if n1 < eps or n2 < eps:
        return np.nan

    cos = np.dot(t_pred, t_gt) / (n1 * n2)
    cos = np.clip(cos, -1.0, 1.0)
    return np.degrees(np.arccos(cos))
