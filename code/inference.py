# %%
import os
import numpy as np
import cv2 
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.ours import Ours_Net
from models.oursmb import Ours_MB
from models.sfpw import SfPW_Net
from models.dsfp import DSfP_Net
from models.unet import UNet
from models.kondo import Kondo_Net

# Choose model and dataset
MODEL_NAME = "ours"  # "ours", "oursmb", "sfpw", "dsfp", "unet", "kondo"
DATASET_NAME = "ThermoPol16" # "ThermoPol16", "Spheres"

EVAL_DATA_ROOT = os.path.join("./data_eval", DATASET_NAME)
SAVE_DIR = os.path.join("./results", DATASET_NAME, MODEL_NAME)


MODEL_DICT = {
    "ours": {
        "model": Ours_Net,
        "weight": "weights/ours.pth"
    },
    "oursmb": {
        "model": Ours_MB,
        "weight": None
    },
    "sfpw": {
        "model": SfPW_Net,
        "weight": "weights/sfpw.pth"
    },
    "dsfp": {
        "model": DSfP_Net,
        "weight": "weights/dsfp.pth"
    },
    "unet": {
        "model": UNet,
        "weight": "weights/unet.pth"
    },
    "kondo": {
        "model": Kondo_Net,
        "weight": "weights/kondo.pth"
    },
}

def main():
    model = MODEL_DICT[MODEL_NAME]["model"](in_channels=8).to(device)
    if MODEL_DICT[MODEL_NAME]["weight"] is not None:
        model.load_state_dict(torch.load(MODEL_DICT[MODEL_NAME]["weight"]))
    model.eval()


    file_list = np.loadtxt(os.path.join(EVAL_DATA_ROOT, "file_list.csv"), dtype=str, delimiter=",", skiprows=1)
    id_list = file_list[:,0]
    mask_list = file_list[:,1]
    stokes_list = file_list[:,3]

    os.makedirs(SAVE_DIR, exist_ok=True)

    for id, mask_path, stokes_path in zip(id_list, mask_list, stokes_list):
        print(f"Processing {id} ...")
        mask = cv2.imread(os.path.join(EVAL_DATA_ROOT, mask_path), cv2.IMREAD_GRAYSCALE) / 255 
        stokes = np.load(os.path.join(EVAL_DATA_ROOT, stokes_path)) 
        
        mask = torch.from_numpy(mask)[None, :, :]
        stokes = torch.from_numpy(stokes)
        tensor = stokes_to_tensor(stokes, mask)
        tensor = tensor[None, :, :, :].float().to(device)

        with torch.no_grad():
            output = model(tensor)
            
        output = output / (torch.linalg.norm(output, dim=1, keepdim=True) + 1e-8)
        output = torch.nan_to_num(output, nan=0.0)
        normal = (output[0].detach().cpu() * mask[0]).permute(1, 2, 0).numpy()
        normal = rotate_normals_view_to_cam(normal)
        normalmap = (normal*0.5+0.5).clip(0,1)

        cv2.imwrite(os.path.join(SAVE_DIR, f"{id}_estimated.png"), (normalmap*65535).astype(np.uint16)[:,:,::-1])


# Stokes to input tensor
def stokes_to_tensor(stokes, mask):
    pol1 = (stokes[0] + stokes[1]) / 2
    pol2 = (stokes[0] - stokes[1]) / 2
    pol3 = (stokes[0] + stokes[2]) / 2
    pol4 = (stokes[0] - stokes[2]) / 2
    intensity = stokes[0]
    dolp = torch.sqrt(stokes[1]**2 + stokes[2]**2) / (stokes[0] + 1e-6) * 10
    aolp1 = stokes[1] / torch.sqrt(stokes[1]**2 + stokes[2]**2 + 1e-6)
    aolp2 = stokes[2] / torch.sqrt(stokes[1]**2 + stokes[2]**2 + 1e-6)
    tensor = torch.stack([pol1, pol2, pol3, pol4, intensity, dolp, aolp1, aolp2], dim=0)
    tensor = tensor * mask
    return tensor

# perspective correction
FOCAL_LENGTH = 320 / np.tan(np.deg2rad(12)) # Horizontal 640pix, HorizontalFoV 24deg
IMAGE_SIZE = 360
def rotate_normals_view_to_cam(projected_normal_vecs):
    h, w, _ = projected_normal_vecs.shape
    if h != w:
        raise ValueError("Only square images are supported now.")
    x_min, x_max = -IMAGE_SIZE/2, IMAGE_SIZE/2
    y_min, y_max = -IMAGE_SIZE/2, IMAGE_SIZE/2

    # define view vector
    view_vecs = np.zeros(shape=(h,w,3), dtype=np.float32)
    view_vecs[:,:,0] = np.linspace(x_min, x_max, w)[:,None]
    view_vecs[:,:,1] = np.linspace(y_min, y_max, h)[None,:]
    view_vecs[:,:,2] = FOCAL_LENGTH
    view_vecs = view_vecs / (np.linalg.norm(view_vecs, axis=2, keepdims=True) + 1e-8)

    # Quick creation of an orthonormal basis
    a = 1 / (1 + view_vecs[:,:,2])
    b = -view_vecs[:,:,0] * view_vecs[:,:,1] * a
    b1 = np.stack([1 - view_vecs[:,:,0] ** 2 * a, b, -view_vecs[:,:,0]], axis=2)
    b2 = np.stack([b, 1 - view_vecs[:,:,1] ** 2 * a, -view_vecs[:,:,1]], axis=2)
    b3 = view_vecs

    # Project the view-based normals back to the camera coordinate system
    normal_vecs = projected_normal_vecs[:,:,0:1] * b1 + \
                  projected_normal_vecs[:,:,1:2] * b2 + \
                  projected_normal_vecs[:,:,2:3] * b3

    return normal_vecs

if __name__ == "__main__":
    main()
# %%
