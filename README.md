# Shape from Polarization of Thermal Emission and Reflection

This is the official implementation of our paper presented at ICCP 2025.

**[[Project Page]](https://k2zm.github.io/SfPTER-ICCP2025/)** | **[[Paper]](https://arxiv.org/pdf/2506.18217)**

Authors: Kazuma Kitazawa, Tsuyoshi Takatani

## Abstract

Shape estimation for transparent objects is challenging due to their complex light transport. To circumvent these difficulties, we leverage the Shape from Polarization (SfP) technique in the Long-Wave Infrared (LWIR) spectrum, where most materials are opaque and emissive. While a few prior studies have explored LWIR SfP, these attempts suffered from significant errors due to inadequate polarimetric modeling, particularly the neglect of reflection. Addressing this gap, we formulated a polarization model that explicitly accounts for the combined effects of emission and reflection. Based on this model, we estimated surface normals using not only a direct model-based method but also a learning-based approach employing a neural network trained on a physically-grounded synthetic dataset. Furthermore, we modeled the LWIR polarimetric imaging process, accounting for inherent systematic errors to ensure accurate polarimetry. We implemented a prototype system and created ThermoPol, the first real-world benchmark dataset for LWIR SfP. Through comprehensive experiments, we demonstrated the high accuracy and broad applicability of our method across various materials, including those transparent in the visible spectrum.

## Installation

We recommend using Python 3.10 or later. You can create a virtual environment and install the required packages using [uv](https://github.com/astral-sh/uv).

```bash
# Create a virtual environment
$ uv venv

# Install dependencies from pyproject.toml
$ uv sync

# Activate the environment
# On macOS/Linux
$ source .venv/bin/activate
# On Windows
> .venv\Scripts\activate
```

## Dataset and Pretrained Weights

You can download the datasets and pretrained weights from the [**Releases**](#).

  * **`data_eval.zip`**: Contains the evaluation datasets `ThermoPol16` and `ThermoPolSpheres` (Stokes vectors, ground truth surface normals, masks).
  * **`data_train.zip`**: Contains the synthetic training dataset `ThermoPolSynth` (Stokes vectors, surface normals, masks, etc.).
  * **`data_raw.zip`**: Contains the raw images for `ThermoPol16` and `ThermoPolSpheres` (TIFF images of the scene and the blackbody shutter).
  * **`weights.zip`**: Contains pretrained weights for our network (`ours.pth`), and for other models (`dsfp.pth`, `sfwp.pth`, `kondo.pth`, `unet.pth`) trained on our `ThermoPolSynth` dataset.

You can also generate your own synthetic dataset using [mitsuba-polarized-emission](https://github.com/k2zm/mitsuba-polarized-emission), our Mitsuba plugin for simulating polarized thermal emission and reflection.

## Usage

First, download and unzip the necessary datasets and weights. Place them in the root directory of this project as shown below:

```
.
├── code/
│   ├── inference.py
│   ├── evaluate.py
│   ├── train.py
│   └── preprocess.py
├── data_eval/
├── data_train/
├── data_raw/
├── weights/
└── ... 
```

To estimate surface normals from Stokes vectors using a pretrained model, run `inference.py`.
``` bash
$ code/inference.py
``` 

To calculate the angular error between the estimated normals and the ground truth, run `evaluate.py`.
```bash
$ code/evaluate.py`  
```

To calculate Stokes vectors from the raw TIFF images, run `preprocess.py`. 

```bash
$ code/preprocess.py
```

To train a new model on the synthetic dataset, run `train.py`.

```bash
$ code/train.py
``` 



## BibTeX

```
@inproceedings{kitazawa2025shape,
  title={Shape from Polarization of Thermal Emission and Reflection},
  author={Kitazawa, Kazuma and Takatani, Tsuyoshi},
  booktitle={Proceedings of the IEEE International Conference on Computational Photography (ICCP)},
  pages={171--181},
  year={2025},
  organization={IEEE}
}
```