# %%
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Choose model and dataset
MODEL_NAME = "ours"  # "ours", "oursmb", "sfpw", "dsfp", "unet", "kondo"
DATASET_NAME = "ThermoPol16" # "ThermoPol16", "Spheres"

ESTIMATED_DIR = os.path.join("./results", DATASET_NAME, MODEL_NAME)
EVAL_DATA_ROOT = os.path.join("./data_eval", DATASET_NAME)

def main():
    # load file list
    file_list = np.loadtxt(os.path.join(EVAL_DATA_ROOT, "file_list.csv"), dtype=str, delimiter=",", skiprows=1)
    id_list = file_list[:,0]
    normal_list = file_list[:,2]

    # load estimated normal maps
    estimated_list = [f"{id}_estimated.png" for id in id_list]

    # make directory for evaluation results
    result_eval_dir = os.path.join(ESTIMATED_DIR, "evaluation")
    os.makedirs(result_eval_dir, exist_ok=True)

    eval_list = []
    for id, normal_path, estimated_path in zip(id_list, normal_list, estimated_list):
        print(f"Processing {id} ...")
        reference_normalmap = cv2.imread(os.path.join(EVAL_DATA_ROOT, normal_path), cv2.IMREAD_UNCHANGED)
        estimated_normalmap = cv2.imread(os.path.join(ESTIMATED_DIR, estimated_path), cv2.IMREAD_UNCHANGED)

        reference_normalmap = cv2.cvtColor(reference_normalmap, cv2.COLOR_BGR2RGB)
        estimated_normalmap = cv2.cvtColor(estimated_normalmap, cv2.COLOR_BGR2RGB)

        reference_normal, reference_mask = normalmap2normal(reference_normalmap)
        estimated_normal, estimated_mask = normalmap2normal(estimated_normalmap)

        angular_error = calc_angular_error(reference_normal, estimated_normal)
        mask = (reference_mask * estimated_mask).clip(0,1)

        # calculate mean and median of angular error
        mean_angular_error = np.sum(angular_error * mask) / np.sum(mask)
        median_angular_error = np.median(angular_error[mask > 0.5])
        rmse = np.sqrt(np.sum((angular_error * mask)**2) / np.sum(mask))
        # calculate accuracy
        acc_1125 = calc_accuracy(angular_error, mask, threshold=11.25)
        acc_2250 = calc_accuracy(angular_error, mask, threshold=22.5)
        acc_3000 = calc_accuracy(angular_error, mask, threshold=30)
        # add results to list
        eval_list.append([mean_angular_error, median_angular_error, rmse, acc_1125, acc_2250, acc_3000])
        
        plt.imsave(os.path.join(result_eval_dir, f"{id}_error.png"), angular_error*mask, cmap='magma', vmin=0, vmax=40)


    eval_list = np.array(eval_list)
    mean_result = np.mean(eval_list, axis=0)
    print("Mean Angular Error: ", mean_result[0])
    print("Median Angular Error: ", mean_result[1])
    print("RMSE: ", mean_result[2])
    print("Accuracy (11.25 deg): ", mean_result[3])
    print("Accuracy (22.5 deg): ", mean_result[4])
    print("Accuracy (30 deg): ", mean_result[5])

    np.savetxt(os.path.join(result_eval_dir, "eval_results.csv"), eval_list, delimiter=",", fmt="%.6f",
            header="Mean Angular Error, Median Angular Error, RMSE, Accuracy (11.25 deg), Accuracy (22.5 deg), Accuracy (30 deg)")
    np.savetxt(os.path.join(result_eval_dir, "mean_eval_results.csv"), mean_result[None, :], delimiter=",", fmt="%.6f",
            header="Mean Angular Error, Median Angular Error, RMSE, Accuracy (11.25 deg), Accuracy (22.5 deg), Accuracy (30 deg)")  


def normalmap2normal(normalmap, max_value=65535):
    normal = np.zeros((normalmap.shape[0], normalmap.shape[1], 3))
    normal[:, :, 0] = normalmap[:, :, 0] / max_value * 2 - 1
    normal[:, :, 1] = normalmap[:, :, 1] / max_value * 2 - 1
    normal[:, :, 2] = normalmap[:, :, 2] / max_value * 2 - 1

    norm = np.linalg.norm(normal, axis=2)
    normal = normal / norm[:, :, None]

    return normal, norm

def calc_angular_error(normal1, normal2):
    # calculate angular error map
    dot_product = np.sum(normal1 * normal2, axis=2)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angular_error = np.arccos(dot_product)
    angular_error = np.rad2deg(angular_error)
    return angular_error

def calc_accuracy(angular_error, mask, threshold=5):
    valid_data = angular_error[mask > 0.5].flatten()
    acc = np.sum(valid_data < threshold) / len(valid_data) * 100
    return acc

if __name__ == "__main__":
    main()