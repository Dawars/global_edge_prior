import os
import time
import yaml
import numpy as np
import pycolmap
from pathlib import Path

from src.utils import *
from src.sfm_utils import *


def filter_sparse_gt_images(sparse_gt, valid_image_names):

    hold_out_image_ids = [
        image.image_id
        for image in sparse_gt.images.values()
        if image.name not in valid_image_names
    ]
    for image_id in hold_out_image_ids:
        del sparse_gt.images[image_id]
    return sparse_gt


def run_scene_reconstruction(args, workspace_path, image_path, scene=None):

    start = time.time()
    camera_prior_sparse_gt = None # no prior is used

    sparse_path = run_reconstruction(
        input_args=args,
        workspace_path=workspace_path,
        image_path=image_path,
        camera_prior_sparse_gt=camera_prior_sparse_gt,
        scene=scene
    )
    reconstruction_time = time.time() - start
    return sparse_path, reconstruction_time


def write_scene_result(
    auc_path,
    dataset_name,
    scene_name,
    args,
    auc_values,
    num_registered,
    num_total,
    reconstruction_time,
    median_error,
):
    with open(auc_path, "a") as file:
        file.write(
            f"{dataset_name} {scene_name} {args.sp_lg} "
            f"{auc_values[0]} {auc_values[1]} {auc_values[2]} {auc_values[3]} "
            f"{num_registered} {num_total} {reconstruction_time * 1000} ms {median_error}\n"
        )


def write_summary_result(auc_path, dataset_name, args, avg_auc):
    with open(auc_path, "a") as file:
        file.write(
            f"{dataset_name} avg {args.sp_lg} "
            f"{avg_auc[0]} {avg_auc[1]} {avg_auc[2]} {avg_auc[3]}\n"
        )


def evaluate_imc(args, year, gt_position_accuracy=0.02):
    error_thresholds = get_error_thresholds(args)
    args.data_path = Path(args.data_path)

    folder_name = f"imc{year}"
    results = {}

    train_root = args.data_path / folder_name / "train"
    for category_path in train_root.iterdir():
        if not category_path.is_dir():
            continue
        if args.categories and category_path.name not in args.categories:
            continue

        category = category_path.name
        results[category] = {}

        print(f"Processing IMC {year}: category={category}")
        all_errors = []
        auc_path = None

        for scene_path in category_path.iterdir():
            if not scene_path.is_dir():
                continue

            scene = scene_path.name
            if args.scenes and scene not in args.scenes:
                continue

            print(f"Processing IMC {year}: category={category}, scene={scene}")

            workspace_path = Path(args.workspace_path) / folder_name / category / scene
            auc_path = workspace_path.parent / ("auc_trees.txt" if args.trees else "auc_knn.txt")

            image_path = scene_path / "images"
            train_image_names = {image.name for image in image_path.iterdir()}
            sparse_gt_path = scene_path / "sfm"

            try:
                sparse_gt = pycolmap.Reconstruction(sparse_gt_path)
            except Exception as e:
                print(f"Failed to load {sparse_gt_path}: {e}")
                continue

            sparse_gt = filter_sparse_gt_images(sparse_gt, train_image_names)
            args.scene_category_path = train_root/ category_path
            args.work_category_path = workspace_path/ category_path

            try:
                sparse_path, reconstruction_time = run_scene_reconstruction(
                    args=args,
                    workspace_path=workspace_path,
                    image_path=image_path,
                    scene=scene
                )
            except Exception as e:
                print(f"Failed on scene {scene}: {e}")
                continue

            components = os.listdir(sparse_path)
            dt_dict = {}
            dr_dict = {}
            all_imgs = set()

            for com in components:
                sparse_path_com = sparse_path / com

                try:
                    sparse_recon = pycolmap.Reconstruction(sparse_path_com)
                except Exception as e:
                    print(f"Failed to load component {sparse_path_com}: {e}")
                    continue

                dts, dRs, pairs = compute_rel_errors(
                    sparse_gt=sparse_gt,
                    sparse=sparse_recon,
                    min_proj_center_dist=gt_position_accuracy,
                    return_pairs=True,
                )

                dt_dict[com] = dts
                dr_dict[com] = dRs

                with open(workspace_path / f"{com}_errors.txt", "w") as file:
                    for i, pair in enumerate(pairs):
                        name0, name1 = pair.split("-")
                        file.write(f"{name0} {name1} {dRs[i]:.2f} {dts[i]:.2f}\n")

                all_imgs.update(img.name for img in sparse_recon.images.values())

            if len(dt_dict) == 0:
                print(f"No valid components for scene {scene}")
                continue

            dts = np.concatenate(list(dt_dict.values())).reshape(len(dt_dict), -1).min(0)
            dRs = np.concatenate(list(dr_dict.values())).reshape(len(dr_dict), -1).min(0)
            errors = [max(dt, dR) for dt, dR in zip(dts, dRs)]

            all_errors.extend(errors)
            results[category][scene] = compute_auc(
                errors,
                error_thresholds,
                min_error=gt_position_accuracy,
            )

            write_scene_result(
                auc_path=auc_path,
                dataset_name=category,
                scene_name=scene,
                args=args,
                auc_values=results[category][scene],
                num_registered=len(all_imgs),
                num_total=len(train_image_names),
                reconstruction_time=reconstruction_time,
                median_error=np.median(np.array(errors)),
            )

        if len(all_errors) == 0:
            print(f"No valid scenes for category {category}")
            continue

        results[category]["__avg__"] = compute_avg_auc(results[category])

        if auc_path is not None:
            write_summary_result(
                auc_path=auc_path,
                dataset_name=category,
                args=args,
                avg_auc=results[category]["__avg__"],
            )

    return results


def evaluate_megadepth(args, gt_position_accuracy=0.02):
    with open("data_dirs.yaml", "r") as f:
        data_dirs = yaml.safe_load(f)

    error_thresholds = get_error_thresholds(args)

    if not args.scenes:
        args.scenes = ["0015", "0022"]

    if set(args.scenes).issubset({"0015", "0022"}):
        args.data_path = Path(data_dirs['megadepth_test_src'])
    else:
        args.data_path = Path(data_dirs['megadepth_train_src'])

    folder_name = "megadepth"
    results = {}
    all_errors = []
    auc_summary_path = None

    for scene_path in args.data_path.iterdir():
        if not scene_path.is_dir():
            continue
        if args.scenes and scene_path.name not in args.scenes:
            continue

        scene = scene_path.name
        print(f"Processing Megadepth: scene={scene}")

        workspace_path = Path(args.workspace_path) / folder_name / scene
        auc_path = workspace_path.parent / ("auc_trees.txt" if args.trees else "auc_knn.txt")
        auc_summary_path = auc_path

        image_path = scene_path
        train_image_names = {image.name for image in image_path.iterdir()}
        sparse_gt_path = Path(data_dirs["megadepth_sfm"]) / scene / "sparse"

        try:
            sparse_gt = pycolmap.Reconstruction(sparse_gt_path)
        except Exception as e:
            print(f"Failed to load {sparse_gt_path}: {e}")
            continue

        sparse_gt = filter_sparse_gt_images(sparse_gt, train_image_names)

        try:
            sparse_path, reconstruction_time = run_scene_reconstruction(
                args=args,
                workspace_path=workspace_path,
                image_path=image_path,
                scene=scene
            )
        except Exception as e:
            print(f"Failed on scene {scene}: {e}")
            continue

        sparse_path = sparse_path / "0"

        try:
            sparse_recon = pycolmap.Reconstruction(sparse_path)
        except Exception as e:
            print(f"Failed to load reconstructed sparse model {sparse_path}: {e}")
            continue

        if args.error_type == "relative":
            dts, dRs = compute_rel_errors(
                sparse_gt=sparse_gt,
                sparse=sparse_recon,
                min_proj_center_dist=gt_position_accuracy,
            )
            errors = [max(dt, dR) for dt, dR in zip(dts, dRs)]

        elif args.error_type == "absolute":
            sparse_aligned_path = workspace_path / "sparse_aligned"
            colmap_alignment(
                args=args,
                sparse_path=sparse_path,
                sparse_gt_path=sparse_gt_path,
                sparse_aligned_path=sparse_aligned_path,
                max_ref_model_error=gt_position_accuracy,
            )
            sparse_aligned = (
                pycolmap.Reconstruction(sparse_aligned_path)
                if (sparse_aligned_path / "images.bin").exists()
                else None
            )
            dts, dRs = compute_abs_errors(
                sparse_gt=sparse_gt,
                sparse=sparse_aligned,
            )
            errors = dts
        else:
            raise ValueError(f"Invalid error type: {args.error_type}")

        all_errors.extend(errors)

        results[scene] = {
            "auc": compute_auc(
                errors,
                error_thresholds,
                min_error=gt_position_accuracy,
            ),
            "num_registered": len(sparse_recon.images),
            "num_total": len(train_image_names),
            "reconstruction_time_ms": reconstruction_time * 1000,
            "median_error": np.median(np.array(errors)),
        }

        write_scene_result(
            auc_path=auc_path,
            dataset_name="megadepth",
            scene_name=scene,
            args=args,
            auc_values=results[scene]["auc"],
            num_registered=results[scene]["num_registered"],
            num_total=results[scene]["num_total"],
            reconstruction_time=reconstruction_time,
            median_error=results[scene]["median_error"],
        )

    if len(all_errors) > 0:
        results["__all__"] = compute_auc(
            all_errors,
            error_thresholds,
            min_error=gt_position_accuracy,
        )

        scene_auc_only = {
            scene_name: scene_result["auc"]
            for scene_name, scene_result in results.items()
            if not scene_name.startswith("__")
        }
        results["__avg__"] = compute_avg_auc(scene_auc_only)

        if auc_summary_path is not None:
            write_summary_result(
                auc_path=auc_summary_path,
                dataset_name="megadepth",
                args=args,
                avg_auc=results["__avg__"],
            )

    return results


def format_results(args, results):
    lines = []

    if "imc2023" in results:
        lines.append("=== IMC2023 ===")
        imc_results = results["imc2023"]

        for category, category_results in imc_results.items():
            lines.append(f"[{category}]")

            for scene, scene_result in category_results.items():
                if scene.startswith("__"):
                    continue
                lines.append(
                    f"  {scene}: "
                    f"AUC2.5={scene_result[0]:.4f}, "
                    f"AUC@5={scene_result[1]:.4f}, "
                    f"AUC@10={scene_result[2]:.4f}, "
                    f"AUC@20={scene_result[3]:.4f}"
                )

            if "__all__" in category_results:
                x = category_results["__all__"]
                lines.append(
                    f"  all: "
                    f"AUC={x[0]:.4f}, AUC@5={x[1]:.4f}, AUC@10={x[2]:.4f}, AUC@20={x[3]:.4f}"
                )

            if "__avg__" in category_results:
                x = category_results["__avg__"]
                lines.append(
                    f"  avg: "
                    f"AUC={x[0]:.4f}, AUC@5={x[1]:.4f}, AUC@10={x[2]:.4f}, AUC@20={x[3]:.4f}"
                )

            lines.append("")

    if "megadepth" in results:
        lines.append("=== Megadepth ===")
        md_results = results["megadepth"]

        for scene, scene_result in md_results.items():
            if scene.startswith("__"):
                continue

            auc = scene_result["auc"]
            lines.append(
                f"{scene}: "
                f"AUC@2.5={auc[0]:.4f}, "
                f"AUC@5={auc[1]:.4f}, "
                f"AUC@10={auc[2]:.4f}, "
                f"AUC@20={auc[3]:.4f}, "
                f"registered={scene_result['num_registered']}/{scene_result['num_total']}, "
                f"time={scene_result['reconstruction_time_ms']:.2f} ms, "
                f"median_error={scene_result['median_error']:.4f}"
            )

        if "__all__" in md_results:
            x = md_results["__all__"]
            lines.append(
                f"all: "
                f"AUC={x[0]:.4f}, AUC@5={x[1]:.4f}, AUC@10={x[2]:.4f}, AUC@20={x[3]:.4f}"
            )

        if "__avg__" in md_results:
            x = md_results["__avg__"]
            lines.append(
                f"avg: "
                f"AUC={x[0]:.4f}, AUC@5={x[1]:.4f}, AUC@10={x[2]:.4f}, AUC@20={x[3]:.4f}"
            )

        lines.append("")

    return "\n".join(lines)


def main():
    args = parse_args()
    print(args)

    results = {}

    if "imc2023" in args.datasets:
        results["imc2023"] = evaluate_imc(args, year=2023)

    if "megadepth" in args.datasets:
        results["megadepth"] = evaluate_megadepth(args)

    print("\nResults\n")
    print(format_results(args, results))


if __name__ == "__main__":
    main()