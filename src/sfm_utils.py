import os
import sys
import yaml
import h5py
import copy
import math
import shutil
import sqlite3
import pycolmap
import subprocess
import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy import integrate

with open("./data_dirs.yaml", "r") as f:
    config = yaml.safe_load(f)

if not os.path.exists(config['hloc_path']):
    print("Warnings: No HLoc loaded, SP+LG features cannot be used in COLMAP")
else:
    sys.path.append(config['hloc_path'])
    from hloc import extract_features, match_features

def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        return 2147483647 * image_id2 + image_id1
    else:
        return 2147483647 * image_id1 + image_id2

def recover_database_images_and_ids(database_path):
    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # Recover database images and ids.
    images = {}
    cameras = {}
    cursor.execute("SELECT name, image_id, camera_id FROM images;")
    for row in cursor:
        images[row[0]] = row[1]
        cameras[row[0]] = row[2]

    # Close the connection to the database.
    cursor.close()
    connection.close()

    return images, cameras

def detect_match_features(args, scene, database_path, images, workspace_path):
    """Feature extraction, matching and return a database for SfM."""

    feature_conf = extract_features.confs["superpoint_aachen"]
    feature_path = extract_features.main(feature_conf, Path(args.scene_category_path)/scene/'images', Path(workspace_path), overwrite=False)
    matcher_conf = match_features.confs["superpoint+lightglue"]
    matches_path = match_features.main(matcher_conf, Path(args.pairs), feature_path, matches=Path(workspace_path)/"matches.h5", overwrite=False)
    write_match_features(args, database_path, images, feature_path, matches_path)

    # # Connect to the database.
    # connection = sqlite3.connect(database_path)
    # cursor = connection.cursor()

    # cursor.execute("DELETE FROM keypoints;")
    # cursor.execute("DELETE FROM descriptors;")
    # cursor.execute("DELETE FROM matches;")
    # cursor.execute("DELETE FROM two_view_geometries;")

    # connection.commit()

    # with open(args.pairs, 'r') as f:
    #     raw_pairs = f.readlines()
    # with h5py.File(feature_path, 'r') as features_saved:
    #     for image_name, image_id in tqdm(images.items(), total=len(images.items())):

    #         keypoints = features_saved[image_name]['keypoints'].__array__()
    #         n_keypoints = keypoints.shape[0]

    #         # Keep only x, y coordinates.
    #         keypoints = keypoints[:, : 2]
    #         # Add placeholder scale, orientation.
    #         keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)
    #         keypoints_str = keypoints.tobytes()
    #         cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
    #                        (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))

    # image_pair_ids = set()
    # with h5py.File(matches_path, 'r') as matches_saved:
    #     for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
    #         try:
    #             image_name1, image_name2 = raw_pair.strip('\n').split(' ')
    #             image_id1, image_id2 = images[image_name1], images[image_name2]
    #             try:
    #                 matches = matches_saved[image_name1][image_name2]['matches'].__array__().astype(np.uint32)
    #                 image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
    #                 if image_id1 > image_id2:
    #                     matches = matches[:, [1, 0]]

    #             except:
    #                 matches = matches_saved[image_name2][image_name1]['matches'].__array__().astype(np.uint32)
    #                 image_pair_id = image_ids_to_pair_id(image_id2, image_id1)
    #                 if image_id2 > image_id1:
    #                     matches = matches[:, [1, 0]]

    #             if image_pair_id in image_pair_ids:
    #                 continue
    #             image_pair_ids.add(image_pair_id)

    #             matches_str = matches.tobytes()
    #             cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
    #                         (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
    #         except:
    #             continue

    # # Close the connection to the database.
    # connection.commit()
    # cursor.close()
    # connection.close()
    if args.delete_h5:
        import glob
        for f in glob.glob(f"{workspace_path}/*.h5"):
            os.remove(f)
        print('.h5 removed')


def write_match_features(args, database_path, images, feature_path, matches_path):
    """Write SP+SG features into a database for SfM."""

    # Connect to the database.
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()
    cursor.execute("DELETE FROM keypoints;")
    cursor.execute("DELETE FROM descriptors;")
    cursor.execute("DELETE FROM matches;")
    cursor.execute("DELETE FROM two_view_geometries;")
    connection.commit()

    with open(args.pairs, 'r') as f:
        raw_pairs = f.readlines()
    with h5py.File(feature_path, 'r') as features_saved:
        for image_name, image_id in tqdm(images.items(), total=len(images.items())):

            keypoints = features_saved[image_name]['keypoints'].__array__()
            n_keypoints = keypoints.shape[0]

            # Keep only x, y coordinates.
            keypoints = keypoints[:, : 2]
            # Add placeholder scale, orientation.
            keypoints = np.concatenate([keypoints, np.ones((n_keypoints, 1)), np.zeros((n_keypoints, 1))], axis=1).astype(np.float32)
            keypoints_str = keypoints.tobytes()
            cursor.execute("INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                           (image_id, keypoints.shape[0], keypoints.shape[1], keypoints_str))

    image_pair_ids = set()
    with h5py.File(matches_path, 'r') as matches_saved:
        for raw_pair in tqdm(raw_pairs, total=len(raw_pairs)):
            try:
                image_name1, image_name2 = raw_pair.strip('\n').split(' ')
                image_id1, image_id2 = images[image_name1], images[image_name2]
                try:
                    matches = matches_saved[image_name1][image_name2]['matches'].__array__().astype(np.uint32)
                    image_pair_id = image_ids_to_pair_id(image_id1, image_id2)
                    if image_id1 > image_id2:
                        matches = matches[:, [1, 0]]

                except:
                    matches = matches_saved[image_name2][image_name1]['matches'].__array__().astype(np.uint32)
                    image_pair_id = image_ids_to_pair_id(image_id2, image_id1)
                    if image_id2 > image_id1:
                        matches = matches[:, [1, 0]]

                if image_pair_id in image_pair_ids:
                    continue
                image_pair_ids.add(image_pair_id)

                matches_str = matches.tobytes()
                cursor.execute("INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);",
                            (image_pair_id, matches.shape[0], matches.shape[1], matches_str))
            except:
                continue

    # Close the connection to the database.
    connection.commit()
    cursor.close()
    connection.close()


def geometric_verification(paths, args):
    """cr.

    https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation
    """
    print('Running geometric verification...')
    subprocess.call(['colmap', 'matches_importer',
                     '--database_path', paths.database_path,
                     '--match_list_path', paths.match_list_path,
                     '--match_type', 'pairs'])



def run_reconstruction(
    input_args,
    workspace_path,
    image_path,
    scene,
    camera_prior_sparse_gt=None,
):
    workspace_path.mkdir(parents=True, exist_ok=True)
    sparse_path =  workspace_path / "sparse"
    database_path = workspace_path / "sp_lg.db" if input_args.sp_lg else workspace_path /"database.db"
    if input_args.overwrite_reconstruction and sparse_path.exists():
        shutil.rmtree(sparse_path)
    if input_args.overwrite_database and database_path.exists():
        database_path.unlink()
    if sparse_path.exists() and len(os.listdir(sparse_path))>0:
        print("Skipping reconstruction, as it already exists")
        return sparse_path

    sparse_path.mkdir(exist_ok=True, parents=True)

    # no database
    if (not os.path.exists(database_path)) or (input_args.overwrite_database):

        # SuperPoint + LightGlue
        if input_args.sp_lg:
            args = [
            "colmap",
            "feature_extractor",
            "--image_path",
            str(image_path),
            "--database_path",
            str(database_path),

            ]
            if input_args.glomap and input_args.use_cpu:
                args += ["--FeatureExtraction.use_gpu", str(False)]
                args += ["--FeatureExtraction.num_threads", str(input_args.num_threads)]

            if input_args.image_list_path is not None:
                args += ["--image_list_path", input_args.image_list_path]
            subprocess.check_call(
                args,
                cwd=workspace_path,
            )


            images, _ = recover_database_images_and_ids(str(database_path))

            if input_args.prepared_sp_lg is not None:
                write_match_features(input_args, str(database_path), images,
                                     feature_path=f"{input_args.prepared_sp_lg}/feats-superpoint-n4096-r1024.h5",
                                     matches_path=f"{input_args.prepared_sp_lg}/matches.h5")

            else:
                detect_match_features(input_args, scene, str(database_path), images, workspace_path)

            args = [
                "colmap",
                "matches_importer",
                "--database_path",
                str(database_path),
                "--match_type",
                "pairs",
                "--match_list_path",
                str(input_args.pairs)
                ]
            if input_args.glomap and input_args.use_cpu:
                # if not input_args.use_gpu:
                args += ["--FeatureMatching.use_gpu", str(False)]
            subprocess.check_call(
                    args,
                    cwd=workspace_path,
                )
        else:
            # SIFT feature extraction
            args = [
            "colmap",
            "feature_extractor",
            "--image_path",
            str(image_path),
            "--database_path",
            str(database_path)
            ]
            if input_args.image_list_path is not None:
                args += ["--image_list_path", input_args.image_list_path]
            subprocess.check_call(
                args,
                cwd=workspace_path,
            )
            if camera_prior_sparse_gt is not None:
                print("Setting prior cameras from GT")
                database = pycolmap.Database(str(database_path))# added database_path due to pycolmap version difference
                database.open(str(database_path))

                camera_id_gt_to_camera_id = {}
                for camera_id_gt, camera_gt in camera_prior_sparse_gt.cameras.items():
                    camera_gt.has_prior_focal_length = True
                    camera_id = database.write_camera(camera_gt, 0)# added one bool due to pycolmap version difference
                    camera_id_gt_to_camera_id[camera_id_gt] = camera_id

                images_gt_by_name = {}
                for image_gt in camera_prior_sparse_gt.images.values():
                    images_gt_by_name[image_gt.name] = image_gt

                for image in database.read_all_images():
                    if image.name not in images_gt_by_name:
                        print(
                            f"Not setting prior camera for image {image.name}, "
                            "because it does not exist in GT"
                        )
                        continue
                    image_gt = images_gt_by_name[image.name]
                    camera_id = camera_id_gt_to_camera_id[image_gt.camera_id]
                    image.camera_id = camera_id
                    database.update_image(image)

                database.close()

            # feature matching
            args = [
                    "colmap",
                    "matches_importer",
                    "--database_path",
                    str(database_path),
                    "--match_type",
                    "pairs",
                    "--match_list_path",
                    input_args.pairs
                    ]

            subprocess.check_call(
                args,
                cwd=workspace_path,
            )

    mapper = "global_mapper" if input_args.glomap else "mapper"

    # mapping
    args = [
        "colmap",
        mapper,
        "--image_path",
        str(image_path),
        "--database_path",
        str(database_path),
        "--output_path",
        str(sparse_path)
    ]
    subprocess.check_call(
        args,
        cwd=workspace_path,
    )

    return sparse_path


def normalize_vec(vec, eps=1e-10):
    return vec / max(eps, np.linalg.norm(vec))


def rot_mat_angular_dist_deg(rot_mat1, rot_mat2):
    cos_dist = np.clip(((np.trace(rot_mat1 @ rot_mat2.T)) - 1) / 2, -1, 1)
    return np.rad2deg(math.acos(cos_dist))


def vec_angular_dist_deg(vec1, vec2):
    cos_dist = np.clip(np.dot(normalize_vec(vec1), normalize_vec(vec2)), -1, 1)
    return np.rad2deg(math.acos(cos_dist))


def get_error_thresholds(args):
    if args.error_type == "relative":
        return args.rel_error_thresholds
    elif args.error_type == "absolute":
        return [100 * t for t in args.abs_error_thresholds]
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")


def compute_rel_errors(sparse_gt, sparse, min_proj_center_dist, return_pairs=False):
    """Computes angular relative pose errors across all image pairs.

    Notice that this approach leads to a super-linear decrease in the AUC scores when multiple images fails to register.
    Consider that we have N images in total in a dataset and M images are registered in the evaluated reconstruction. In
    this case, we can compute "finite" errors for (N-M)^2 pairs while the dataset has a total of N^2 pairs. In case of
    many unregistered images, the AUC score will drop much more than the (intuitively) expected (N-M) / N ratio. One
    could appropriately normalize by computing a single score per image through a suitable normalization of all pairwise
    errors per image. However, this becomes difficult when multiple sub-components are incorrectly stitched together in
    the same reconstruction (e.g., in the case of symmetry issues).
    """

    if sparse is None:
        print("Reconstruction failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    if return_pairs: pairs = []
    images = {}
    for image in sparse.images.values():
        images[image.name] = image
    dts = []
    dRs = []

    for this_image_gt in sparse_gt.images.values():
        if this_image_gt.name not in images:
            for _ in range(sparse_gt.num_images() - 1):
                dts.append(np.inf)
                dRs.append(180)
                if return_pairs: pairs.append(f"{this_image_gt.name}-None")

            continue

        this_image = images[this_image_gt.name]

        for other_image_gt in sparse_gt.images.values():
            if this_image_gt.image_id == other_image_gt.image_id:
                continue

            if other_image_gt.name not in images:
                dts.append(np.inf)
                dRs.append(180)
                if return_pairs: pairs.append(f"{this_image_gt.name}-{other_image_gt.name}")

                continue

            other_image = images[other_image_gt.name]

            this_from_other = (
                this_image.cam_from_world * other_image.cam_from_world.inverse()
            )
            this_from_other_gt = (
                this_image_gt.cam_from_world
                * other_image_gt.cam_from_world.inverse()
            )

            proj_center_dist_gt = np.linalg.norm(
                this_image_gt.projection_center()
                - other_image_gt.projection_center()
            )
            if proj_center_dist_gt < min_proj_center_dist:
                # If the cameras almost coincide, then the angular direction
                # distance is unstable, because a small position change can
                # cause a large rotational error. In this case, we only measure
                # rotational relative pose error.
                dt = 0
            else:
                dt = vec_angular_dist_deg(
                    this_from_other.translation, this_from_other_gt.translation
                )

            dR = rot_mat_angular_dist_deg(
                this_from_other.rotation.matrix(),
                this_from_other_gt.rotation.matrix(),
            )
            if return_pairs: pairs.append(f"{this_image.name}-{other_image.name}")
            dts.append(dt)
            dRs.append(dR)
    if return_pairs:
        return dts, dRs, pairs
    else:
        return dts, dRs


def compute_rel_errors_glomap(sparse_gt, sparse, min_proj_center_dist, return_pairs=False):
    """Computes angular relative pose errors across all image pairs.

    Notice that this approach leads to a super-linear decrease in the AUC scores when multiple images fails to register.
    Consider that we have N images in total in a dataset and M images are registered in the evaluated reconstruction. In
    this case, we can compute "finite" errors for (N-M)^2 pairs while the dataset has a total of N^2 pairs. In case of
    many unregistered images, the AUC score will drop much more than the (intuitively) expected (N-M) / N ratio. One
    could appropriately normalize by computing a single score per image through a suitable normalization of all pairwise
    errors per image. However, this becomes difficult when multiple sub-components are incorrectly stitched together in
    the same reconstruction (e.g., in the case of symmetry issues).
    """

    if sparse is None:
        print("Reconstruction failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    if return_pairs: pairs = []
    images = {}
    for image in sparse.images.values():
        images[image.name] = image
    dts = []
    dRs = []

    for this_image_gt in sparse_gt.images.values():
        if this_image_gt.name not in images:
            for _ in range(sparse_gt.num_images() - 1):
                dts.append(np.inf)
                dRs.append(180)
                if return_pairs: pairs.append(f"{this_image_gt.name}-None")

            continue

        this_image = images[this_image_gt.name]

        for other_image_gt in sparse_gt.images.values():
            if this_image_gt.image_id == other_image_gt.image_id:
                continue

            if other_image_gt.name not in images:
                dts.append(np.inf)
                dRs.append(180)
                if return_pairs: pairs.append(f"{this_image_gt.name}-{other_image_gt.name}")

                continue

            other_image = images[other_image_gt.name]

            this_from_other = (
                this_image.cam_from_world() * other_image.cam_from_world().inverse()
            )
            this_from_other_gt = (
                this_image_gt.cam_from_world()
                * other_image_gt.cam_from_world().inverse()
            )

            proj_center_dist_gt = np.linalg.norm(
                this_image_gt.projection_center()
                - other_image_gt.projection_center()
            )
            if proj_center_dist_gt < min_proj_center_dist:
                # If the cameras almost coincide, then the angular direction
                # distance is unstable, because a small position change can
                # cause a large rotational error. In this case, we only measure
                # rotational relative pose error.
                dt = 0
            else:
                dt = vec_angular_dist_deg(
                    this_from_other.translation, this_from_other_gt.translation
                )

            dR = rot_mat_angular_dist_deg(
                this_from_other.rotation.matrix(),
                this_from_other_gt.rotation.matrix(),
            )
            if return_pairs: pairs.append(f"{this_image.name}-{other_image.name}")
            dts.append(dt)
            dRs.append(dR)
    if return_pairs:
        return dts, dRs, pairs
    else:
        return dts, dRs

def intrinsics_from_params(params):
    fx, fy, cx, cy = params  # Ignore k for intrinsic matrix
    K = np.array([
        [fx,  0, cx],
        [0,  fy, cy],
        [0,   0,  1]
    ])
    return K

def compute_abs_errors(sparse_gt, sparse):
    if sparse is None:
        print("Reconstruction or alignment failed")
        return len(sparse_gt.images) * [np.inf], len(sparse_gt.images) * [180]

    images = {}
    for image in sparse.images.values():
        images[image.name] = image

    dts = []
    dRs = []
    for image_gt in sparse_gt.images.values():
        if image_gt.name not in images:
            dts.append(np.inf)
            dRs.append(180)
            continue

        image = images[image_gt.name]

        dt = np.linalg.norm(
            image_gt.projection_center() - image.projection_center()
        )
        dR = rot_mat_angular_dist_deg(
            image_gt.cam_from_world.rotation.matrix(),
            image.cam_from_world.rotation.matrix(),
        )

        dts.append(dt)
        dRs.append(dR)

    return dts, dRs

def compute_recall(errors):
    num_elements = len(errors)
    sort_idx = np.argsort(errors)
    errors = np.array(errors.copy())[sort_idx]
    recall = (np.arange(num_elements) + 1) / num_elements
    return errors, recall


def compute_auc(errors, thresholds, min_error=None):
    if len(errors) == 0:
        raise ValueError("No errors to evaluate")

    errors, recall = compute_recall(errors)

    if min_error is not None:
        min_index = np.searchsorted(errors, min_error, side="right")
        min_score = min_index / len(errors)
        recall = np.r_[min_score, min_score, recall[min_index:]]
        errors = np.r_[0, min_error, errors[min_index:]]
    else:
        recall = np.r_[0, recall]
        errors = np.r_[0, errors]

    aucs = []
    for t in thresholds:
        last_index = np.searchsorted(errors, t, side="right")
        r = np.r_[recall[:last_index], recall[last_index - 1]]
        e = np.r_[errors[:last_index], t]
        auc = integrate.trapezoid(r, x=e) / t
        aucs.append(auc * 100)
    return aucs


def compute_avg_auc(scene_aucs):
    auc_sum = None
    num_scenes = 0
    for scene, aucs in scene_aucs.items():
        if scene.startswith("__") and scene.endswith("__"):
            continue
        num_scenes += 1
        if auc_sum is None:
            auc_sum = copy.copy(aucs)
        else:
            for i in range(len(auc_sum)):
                auc_sum[i] += aucs[i]
    return [auc / num_scenes for auc in auc_sum]



def format_results(args, results):
    if args.error_type == "relative":
        metric = "AUC @ X deg (%)"
        thresholds = args.rel_error_thresholds
    elif args.error_type == "absolute":
        metric = "AUC @ X cm (%)"
        thresholds = [100 * t for t in args.abs_error_thresholds]
    else:
        raise ValueError(f"Invalid error type: {args.error_type}")

    column = "scenes"
    size1 = max(
        len(column) + 2,
        max(len(s) for d in results.values() for c in d.values() for s in c),
    )
    size2 = max(len(metric) + 2, len(thresholds) * 7 - 1)
    header = f"{column:=^{size1}} {metric:=^{size2}}"
    header += "\n" + " " * (size1 + 1)
    header += " ".join(f'{str(t).rstrip("."):^6}' for t in thresholds)
    text = [header]
    for dataset, category_results in results.items():
        for category, scene_results in category_results.items():
            text.append(f"\n{dataset + '=' + category:=^{size1 + size2 + 1}}")
            for scene, aucs in scene_results.items():
                assert len(aucs) == len(thresholds)
                row = ""
                if scene == "__avg__":
                    scene = "average"
                    row += "-" * (size1 + size2 + 1) + "\n"
                if scene == "__all__":
                    scene = "overall"
                    row += "-" * (size1 + size2 + 1) + "\n"
                row += f"{scene:<{size1}} "
                row += " ".join(f"{auc:>6.2f}" for auc in aucs)
                text.append(row)
    return "\n".join(text)


def lamar_recalls(dts, dRs, Rt_thresholds=((1.0, 0.1), (5.0, 1.0))):
    """
    Rt_thresholds: list of (deg, meter)
    """
    dts = np.asarray(dts, dtype=float)
    dRs = np.asarray(dRs, dtype=float)
    recalls = [float(np.mean((dRs < th_r) & (dts < th_t))) for th_r, th_t in Rt_thresholds]
    return recalls



def colmap_alignment(
    args, sparse_path, sparse_gt_path, sparse_aligned_path, max_ref_model_error
):
    if args.overwrite_alignment and sparse_aligned_path.exists():
        shutil.rmtree(sparse_aligned_path)
    if sparse_aligned_path.exists():
        print("Skipping alignment, as it already exists")
        return
    src = pycolmap.Reconstruction(str(sparse_path))
    gt  = pycolmap.Reconstruction(str(sparse_gt_path))
    names = {img.name for img in src.images.values()}
    with open(sparse_path/"ref_subset.txt", "w") as f:
        for img in gt.images.values():
            if img.name in names:
                C = img.projection_center()
                f.write(f"{img.name} {C[0]} {C[1]} {C[2]}\n")

    if sparse_path.exists():
        sparse_aligned_path.mkdir(parents=True, exist_ok=True)
        subprocess.call(
            [
                "colmap",
                "model_aligner",
                "--input_path",
                sparse_path,
                "--ref_images_path",
                sparse_path/'ref_subset.txt',
                "--output_path",
                sparse_aligned_path,
                "--alignment_max_error",
                str(max_ref_model_error),
                "--ref_is_gps", str(0),
                "--min_common_images", "5",
            ]
        )
