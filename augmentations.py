import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AddGaussianNoise(nn.Module):
    def __init__(self, scale: float = 0.1):
        super().__init__()
        assert 0 <= scale <= 1
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.std(x, dim=1, keepdim=True) * self.scale
        noise = (std**0.5) * torch.randn_like(x)
        x = x + noise
        return x


class TranslateToOrigin(nn.Module):
    def __init__(self, joint_idx: int):
        super().__init__()
        self.joint_idx = joint_idx

    def forward(self, skeleton_seq: torch.Tensor):
        M, T, V, C = skeleton_seq.shape
        ref_xy = skeleton_seq[:, 0, self.joint_idx, :2].unsqueeze(1).unsqueeze(2)  # [M, 1, 1, C]

        skeleton_seq_xy = skeleton_seq[..., :2] - ref_xy  # (M, T, V, 2)
        skeleton_seq_z = skeleton_seq[..., 2:].clone()  # (M, T, V, 1)

        return torch.cat((skeleton_seq_xy, skeleton_seq_z), dim=-1)  # (M, T, V, 3)


class AlignShouldersToXAxis(nn.Module):
    def __init__(self, left_shoulder: int, right_shoulder: int):
        """
        Alinha o vetor entre os ombros ao eixo X no primeiro frame.
        """
        super().__init__()
        self.left_shoulder = left_shoulder
        self.right_shoulder = right_shoulder

    def forward(self, skeleton_seq: torch.Tensor):
        """
        skeleton_seq: [T, J, 3] ou [B, T, J, 3]
        """
        return torch.stack(
            [self._align_single_sequence(seq) for seq in skeleton_seq],
            dim=0,
        )

    def _align_single_sequence(self, seq: torch.Tensor):

        p_left = seq[0, self.left_shoulder]
        p_right = seq[0, self.right_shoulder]
        shoulder_vec = p_right - p_left
        shoulder_vec[2] = 0.0

        if torch.norm(shoulder_vec) < 1e-6:
            return seq

        shoulder_vec = F.normalize(shoulder_vec, dim=0)
        target_vec = torch.tensor([1.0, 0.0, 0.0], device=seq.device, dtype=shoulder_vec.dtype)

        axis = torch.cross(shoulder_vec, target_vec)
        angle = torch.acos(torch.clamp(torch.dot(shoulder_vec, target_vec), -1.0, 1.0))

        if torch.norm(axis) < 1e-6 or torch.isnan(angle):
            return seq

        axis = F.normalize(axis, dim=0)

        K = torch.tensor([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], device=seq.device)

        I = torch.eye(3, device=seq.device)
        R = I + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)

        T, V, C = seq.shape
        seq_flat = seq.view(-1, 3)  # [(T*V), 3]
        rotated = torch.matmul(seq_flat, R.T)  # [(T*V), 3]
        return rotated.view(T, V, C)


class RandomRotation3DTransform(nn.Module):
    def __init__(self, max_angle=0.3, axes=("x", "y", "z")):
        super().__init__()
        self.max_angle = max_angle
        self.axes = axes

    def forward(self, data):
        M, T, V, C = data.shape
        data = data.permute(1, 3, 2, 0).contiguous().view(T, C, V * M)  # [T, 3, V*M]
        rot_angles = self._sample_angles(T)  # [T, 3, 3]
        rotated = torch.matmul(rot_angles.to(data.dtype), data)  # [T, 3, V*M]
        rotated = rotated.view(T, C, V, M).permute(3, 0, 2, 1).contiguous()  # [M, T, V, C]
        return rotated

    def _sample_angles(self, T):
        angles = torch.zeros(3)
        if "x" in self.axes:
            angles[0] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        if "y" in self.axes:
            angles[1] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        if "z" in self.axes:
            angles[2] = torch.empty(1).uniform_(-self.max_angle, self.max_angle)
        rot = angles.repeat(T, 1)  # [T, 3]
        return self._compute_rot_matrix(rot)  # [T, 3, 3]

    def _compute_rot_matrix(self, rot):
        cos_r, sin_r = rot.cos(), rot.sin()
        zeros = torch.zeros(rot.shape[0], 1)
        ones = torch.ones(rot.shape[0], 1)

        r1 = torch.stack((ones, zeros, zeros), dim=-1)
        rx2 = torch.stack((zeros, cos_r[:, 0:1], sin_r[:, 0:1]), dim=-1)
        rx3 = torch.stack((zeros, -sin_r[:, 0:1], cos_r[:, 0:1]), dim=-1)
        rx = torch.cat((r1, rx2, rx3), dim=1)  # [T, 3, 3]

        ry1 = torch.stack((cos_r[:, 1:2], zeros, -sin_r[:, 1:2]), dim=-1)
        r2 = torch.stack((zeros, ones, zeros), dim=-1)
        ry3 = torch.stack((sin_r[:, 1:2], zeros, cos_r[:, 1:2]), dim=-1)
        ry = torch.cat((ry1, r2, ry3), dim=1)

        rz1 = torch.stack((cos_r[:, 2:3], sin_r[:, 2:3], zeros), dim=-1)
        r3 = torch.stack((zeros, zeros, ones), dim=-1)
        rz2 = torch.stack((-sin_r[:, 2:3], cos_r[:, 2:3], zeros), dim=-1)
        rz = torch.cat((rz1, rz2, r3), dim=1)

        return rz @ ry @ rx  # [T, 3, 3]


class ScaleHands(nn.Module):
    def __init__(self, left_hand_joints, right_hand_joints, scale_factor=1.5):
        """
        Args:
            left_hand_joints (list[int]): indices of left hand joints
            right_hand_joints (list[int]): indices of right hand joints
            scale_range (tuple): random scale factor range (min, max)
        """
        super().__init__()
        self.left_hand_joints = left_hand_joints
        self.right_hand_joints = right_hand_joints
        self.scale_factor = scale_factor

    def forward(self, skeleton: torch.Tensor) -> torch.Tensor:
        """
        Args:
            skeleton (Tensor): shape (M, T, V, C) or (T, V, C)
        Returns:
            Tensor: same shape as input, with hand joints scaled
        """
        orig_shape = skeleton.shape
        if skeleton.dim() == 3:
            skeleton = skeleton.unsqueeze(0)  # Add M dimension

        M, T, V, C = skeleton.shape

        # Center of the hand (approximation): use wrist joint (e.g., joint 9 and 10)
        wrist_indices = [self.left_hand_joints[0], self.right_hand_joints[0]]

        left_wrist_coords = skeleton[:, :, self.left_hand_joints[0], :].unsqueeze(2)  # shape (M, T, 1, C)
        right_wrist_coords = skeleton[:, :, self.right_hand_joints[0], :].unsqueeze(2)  # shape (M, T, 1, C)

        # Apply scaling around wrist
        coords = skeleton[:, :, self.left_hand_joints, :]
        centered = coords - left_wrist_coords
        scaled = centered * self.scale_factor
        skeleton[:, :, self.left_hand_joints, :] = scaled + left_wrist_coords

        coords = skeleton[:, :, self.right_hand_joints, :]
        centered = coords - right_wrist_coords
        scaled = centered * self.scale_factor
        skeleton[:, :, self.right_hand_joints, :] = scaled + right_wrist_coords

        return skeleton.squeeze(0) if orig_shape != skeleton.shape else skeleton
