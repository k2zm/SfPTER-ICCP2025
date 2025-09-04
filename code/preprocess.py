# %%
import os

import numpy as np
import scipy
import scipy.interpolate 
import cv2
import glob

RAW_ROOT = "./data_raw/raw_ThermoPol16"
SAVE_ROOT = "./data_raw/processed_ThermoPol16"

# calibed values
GAIN = 33.858
BOARD_EMISSIVITY = 0.99433
CAM_ABSORBANSE = 0.049476
SCALE = 7000

ROI_SIZE = 360
RAW_HEIGHT = 512
RAW_WIDTH = 640

os.makedirs(SAVE_ROOT, exist_ok=True)
roi = (RAW_HEIGHT//2-ROI_SIZE//2, ROI_SIZE, RAW_WIDTH//2-ROI_SIZE//2, ROI_SIZE)

def main():
    file_list = np.loadtxt(os.path.join(RAW_ROOT, "file_list.csv"), dtype=str, delimiter=",", skiprows=1)
    ids = file_list[:, 0]
    dir_names = file_list[:, 1]
    black_temps = file_list[:, 2].astype(np.float32)

    # for i in range(len(ids)):
    for i, (id, dir_name, black_temp) in enumerate(zip(ids, dir_names, black_temps)):
        directory = os.path.join(RAW_ROOT, dir_name)
        print(f"Processing {ids[i]} ...")

        images, ts = load_diff_images_with_time(directory)
        thetas = ts / 1200 * 180    # 1200s per 180deg rotation
        in_range = np.logical_and(0 <= thetas, thetas <= 180)
        images = images[in_range]
        ts = ts[in_range]
        thetas = thetas[in_range]

        transmittance = theta_to_transmittance(thetas, CAM_ABSORBANSE / 2)
        images = images / transmittance[:, None, None]
        images = images + GAIN * temp_to_emission(black_temp) / 2 * BOARD_EMISSIVITY

        stokes = continuous_polimages_to_stokes(images, np.deg2rad(thetas))

        cropped_stokes = stokes[:, roi[0]:roi[0] + roi[1], roi[2]:roi[2] + roi[3]]
        cropped_stokes = cropped_stokes / SCALE

        intensity = cropped_stokes[0, :, :]
        intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))

        os.makedirs(SAVE_ROOT, exist_ok=True)
        np.save(os.path.join(SAVE_ROOT, f"{dir_name}_{id}_stokes.npy"), cropped_stokes)
        cv2.imwrite(os.path.join(SAVE_ROOT, f"{dir_name}_{id}_intensity.png"), (intensity * 255).astype(np.uint8))


def continuous_polimages_to_stokes(intensities, polarizer_angles):
    """
    Calculate only linear polarization stokes parameters from measured intensities and linear polarizer angle.
    Minimal implementation for intensities(n,h,w) and polarizer_angles(n).

    Parameters
    ----------
    intensities : ndarray(n,h,w)
        Intensities for each polarizer angle.
    polarizer_angles : ndarray(n)
        Polarizer angles in radian.

    Returns
    -------
    stokes : ndarray(3,h,w)
        Calculated stokes parameters (S0, S1, S2).
    """
    intensities = np.array(intensities, dtype=np.float32)
    polarizer_angles = np.array(polarizer_angles, dtype=np.float32)
    assert intensities.shape[0] == polarizer_angles.shape[0]
    n, h, w = intensities.shape

    # Construct observation matrix A (n, 3)
    A = np.zeros((n, 3))
    for i in range(n):
        angle = polarizer_angles[i]
        A[i, 0] = 1.0
        A[i, 1] = -np.cos(2 * angle)
        A[i, 2] = np.sin(2 * angle)

    # Reshape intensities to (n, h*w) to process all pixels at once
    intensities_reshaped = intensities.reshape(n, h * w)

    # Calculate pseudo-inverse of A: (3, n)
    A_pinv = np.linalg.pinv(A)

    # Calculate stokes parameters for all pixels: (3, h*w)
    stokes_reshaped = A_pinv @ intensities_reshaped

    # Reshape stokes back to (3, h, w)
    stokes = stokes_reshaped.reshape(3, h, w)

    return stokes*2

def temp_to_emission(temp):
    # Stefan–Boltzmann law for radiance (W * sr^-1 * m^-2)
    return (temp + 273.15)**4 * 1.804e-8

def theta_to_transmittance(x, amplitude):
    return amplitude * np.cos(np.deg2rad(x-8) * 2) + (1 - amplitude)

def crop_center(img, crop_width, crop_height):
    height, width = img.shape[:2]
    
    start_x = max(0, int((width - crop_width) / 2))
    start_y = max(0, int((height - crop_height) / 2))    
    end_x = min(width, start_x + crop_width)
    end_y = min(height, start_y + crop_height)
    cropped_img = img[start_y:end_y, start_x:end_x]
    
    return cropped_img

def sort_filepaths_numerically(filepaths):
    def numeric_key(filepath):
        try:
            filename = os.path.basename(filepath).split(".")[0]
            return int(filename)
        except ValueError:
            return float('inf')
    return sorted(filepaths, key=numeric_key)

def load_images(fileroot):
    scene_root = os.path.join(fileroot, "scene")
    black_root = os.path.join(fileroot, "black")

    scene_image_paths = glob.glob(os.path.join(scene_root, "*.tif"))
    scene_image_paths = sort_filepaths_numerically(scene_image_paths)
    black_image_paths = glob.glob(os.path.join(black_root, "*.tif"))
    black_image_paths = sort_filepaths_numerically(black_image_paths)

    raw_scene_images = []
    for scene_image_path in scene_image_paths:
        scene_image = cv2.imread(scene_image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        raw_scene_images.append(scene_image)
    raw_scene_images = np.stack(raw_scene_images).astype(np.float32)

    raw_black_images = []
    for black_image_path in black_image_paths:
        black_image = cv2.imread(black_image_path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)
        raw_black_images.append(black_image)
    raw_black_images = np.stack(raw_black_images).astype(np.float32)

    scene_indices = [os.path.basename(scene_image_path) for scene_image_path in scene_image_paths]
    scene_indices = [int(os.path.splitext(index)[0]) for index in scene_indices]
    scene_indices = np.array(scene_indices, dtype=np.float32)

    black_indices = [os.path.basename(black_image_path) for black_image_path in black_image_paths]
    black_indices = [int(os.path.splitext(index)[0]) for index in black_indices]
    black_indices = np.array(black_indices, dtype=np.float32)

    return raw_scene_images, raw_black_images, scene_indices, black_indices


def load_diff_images_with_time(fileroot):
    raw_scene_images, raw_black_images, scene_ts, black_ts = load_images(fileroot)

    interp = scipy.interpolate.interp1d(black_ts, raw_black_images, axis=0, fill_value="extrapolate")
    raw_black_images_interp = interp(scene_ts)
    
    diff_images = raw_scene_images - raw_black_images_interp
    return diff_images, scene_ts

if __name__ == "__main__":
    main()
    
# %%
