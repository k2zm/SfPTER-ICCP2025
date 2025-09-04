# Our model-based method
# No learnalbe parameters, using PyTorch just for interface compatibility.

import numpy as np
import cv2
import torch
import torch.nn as nn
import math

SOBEL_KERNEL_SIZE = 3
ETA = 1.8
I_T = 1.0
I_R = 0.7
ZENITH_MAX = 78

class Ours_MB(nn.Module):
    def __init__(self, in_channels, dim=32):
        super().__init__()

        zenith_to_dolp = get_zenith_to_dolp(ETA, I_R, I_T)
        zenith_min = 0
        zenith_max = np.deg2rad(ZENITH_MAX)
        self.sfp = SfP(zenith_to_dolp, zenith_min, zenith_max)

    def forward(self, x):
        b, c, h, w = x.size()
        assert b == 1, "Batch size must be 1."

        intensity = x[:,4,:,:]
        dolp10x = x[:,5,:,:]
        aolp1 = x[:,6,:,:]
        aolp2 = x[:,7,:,:]

        mask = (intensity > 0).float()
        dolp = dolp10x / 10
        aolp = (torch.atan2(aolp2, aolp1) % (2 * math.pi)) / 2

        normal = self.sfp(mask[0].cpu().numpy(), dolp[0].cpu().numpy(), aolp[0].cpu().numpy())
        normal = torch.tensor(normal).permute(2, 0, 1).unsqueeze(0)
        return normal
        

def get_zenith_to_dolp(eta, I_R, I_T):
    def zenith_to_dolp(zenith):
        # 光線の入射角
        theta_i_R = zenith
        theta_i_T = np.arcsin(np.sin(zenith) / eta)

        # 反射率
        F_p_R = (np.tan(theta_i_R - theta_i_T) / (np.tan(theta_i_R + theta_i_T)+1e-8))**2
        F_s_R = (np.sin(theta_i_R - theta_i_T) / (np.sin(theta_i_R + theta_i_T)+1e-8))**2

        # 透過率
        F_p_T = (1 - F_p_R)
        F_s_T = (1 - F_s_R)

        # p偏光とs偏光の光強度
        I_p = I_R * F_p_R + I_T * F_p_T
        I_s = I_R * F_s_R + I_T * F_s_T
        I_min = I_s / (I_s + I_p)
        I_max = I_p / (I_s + I_p)
        dolp = (I_max - I_min) / (I_max + I_min)
        return dolp
    return zenith_to_dolp


class SfP:
    """
    偏光度と偏光角から法線方向を推定するクラス
    1. 偏光度から天頂角を計算
    2. 偏光角から方位角候補を計算
    3. 外縁部から順に方位角を決定

    
    近傍法線に近い候補を選択するように，外縁部から1ピクセル幅ずつ法線を決定する
    そのたびに更新予定ピクセルだけでなく全ピクセルで近傍法線を計算している（なんて非効率！）
    ただし実装が簡単なので，とりあえずOK（そんなにおそくない）
    """
    def __init__(self, zenith_to_dolp, zenith_min=0, zenith_max=np.pi/2):
        self.zenith_array = np.linspace(zenith_min, zenith_max, 1000)
        self.dolp_array = zenith_to_dolp(self.zenith_array)

        if not np.all(np.diff(self.dolp_array) >= 0):
            raise ValueError('zenith_to_dolp は単調増加である必要があります')

    def dolp_to_zenith(self, dolp):
        """
        dolp (偏光度) から zenith (天頂角) を計算する関数
        """
        zenith = np.interp(dolp, self.dolp_array, self.zenith_array)
        return zenith

    def aolp_to_azimuth(self, aolp):
        """
        aolp (偏光角) から azimuth (方位角) を計算する関数
        """
        azimuth = aolp
        return azimuth

    # =============================================================================
    # 外縁部から順に法線方位角を決定
    # =============================================================================

    def __call__(self, mask, dolp, aolp):
        """
        外縁部から順に法線方位角を決定
        """
        zenith = self.dolp_to_zenith(dolp)
        azimuth_cand = self.aolp_to_azimuth(aolp)
        normal = np.zeros((mask.shape[0], mask.shape[1], 3))

        dist_from_edge = cv2.distanceTransform(np.where(mask>0, 255, 0).astype(np.uint8), cv2.DIST_L1, 5)
        n_iter = dist_from_edge.max().astype(np.int32)

        # 物体領域外縁の法線を決定
        target_pixels_1 = (dist_from_edge == 1)
        azimuth_1 = self.edge_azimuth(mask, azimuth_cand)
        normal[target_pixels_1] = np.array([np.sin(zenith[target_pixels_1]) * np.cos(azimuth_1[target_pixels_1]),
                                            np.sin(zenith[target_pixels_1]) * np.sin(azimuth_1[target_pixels_1]),
                                            np.cos(zenith[target_pixels_1])]).T

        # 物体領域内部の法線を決定
        for i in range(2, n_iter+1):
            target_pixels_i = (dist_from_edge == i)
            azimuth_i = self.inner_azimuth(normal, azimuth_cand)
            normal[target_pixels_i] = np.array([np.sin(zenith[target_pixels_i]) * np.cos(azimuth_i[target_pixels_i]),
                                                np.sin(zenith[target_pixels_i]) * np.sin(azimuth_i[target_pixels_i]),
                                                np.cos(zenith[target_pixels_i])]).T

        return normal
    
    def azimuth_correct(self, azimuth_candidate, neighbor_azimuth):
        """
        2π周期の角度の差を計算し，候補と隣接平均とのずれが π/2 を超える場合は π を加算して曖昧性を解消
        """
        # 角度差を 0~π の範囲に正規化
        diff = np.abs(azimuth_candidate - neighbor_azimuth) % (2 * np.pi)
        diff = np.minimum(diff, 2 * np.pi - diff)
        # 差が π/2 を超えていたら π を足して補正
        corrected = np.where(diff > (np.pi/2), azimuth_candidate + np.pi, azimuth_candidate)
        return corrected % (2 * np.pi)

    def edge_azimuth(self, mask, azimuth_cand):
        """
        マスク外縁の勾配に基づき方位角を決定
        """
        # エッジの勾配方向を計算
        grad_x = -cv2.Sobel(mask, cv2.CV_64F, 1, 0, ksize=SOBEL_KERNEL_SIZE)
        grad_y = cv2.Sobel(mask, cv2.CV_64F, 0, 1, ksize=SOBEL_KERNEL_SIZE)
        grad_norm = np.sqrt(grad_x**2 + grad_y**2) + 1e-8  # 0div防止
        unit_grad_x = grad_x / grad_norm
        unit_grad_y = grad_y / grad_norm
        grad_azimuth = np.arctan2(unit_grad_y, unit_grad_x)
        
        # 候補方位角とエッジ勾配との整合性で ambiguity を解消
        return self.azimuth_correct(azimuth_cand, grad_azimuth)

    def inner_azimuth(self, normal, azimuth_cand):
        """
        近傍ピクセルの法線に基づき方位角を決定
        """
        # 周辺ピクセルの法線の平均ベクトルを計算
        normal_neighbor = cv2.blur(normal, (SOBEL_KERNEL_SIZE, SOBEL_KERNEL_SIZE))
        normal_neighbor = normal_neighbor / np.linalg.norm(normal_neighbor, axis=-1, keepdims=True)
        normal_neighbor_azimuth = np.arctan2(normal_neighbor[..., 1], normal_neighbor[..., 0])

        return self.azimuth_correct(azimuth_cand, normal_neighbor_azimuth)