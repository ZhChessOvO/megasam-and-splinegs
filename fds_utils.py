import torch
import torch.nn.functional as F
from raft.raft import RAFT
from raft.utils.utils import InputPadder
from typing import Tuple, List

class FDSUtils:
    def __init__(self, raft_model: RAFT):
        """
        初始化FDS工具类
        :param raft_model: 预训练的RAFT模型（eval模式）
        """
        self.raft = raft_model
        self.device = next(raft_model.parameters()).device  # 获取模型所在设备

    def generate_perturbed_pose(self, w2c: torch.Tensor) -> torch.Tensor:
        """
        生成随机扰动的相机位姿（世界到相机的4x4变换矩阵）
        :param w2c: 原始相机位姿，形状为(4, 4)或(batch_size, 4, 4)
        :return: 扰动后的相机位姿，形状与输入一致
        """
        # TODO: 后续补充随机扰动逻辑（参考FDS的fds_sample方法）
        # 暂时直接返回原始位姿
        return w2c.clone()

    def compute_radiance_flow(self, 
                             w2c1: torch.Tensor, 
                             w2c2: torch.Tensor, 
                             gaussians: dict, 
                             K: torch.Tensor,
                             image_size: Tuple[int, int]) -> torch.Tensor:
        """
        根据两个相机位姿和高斯点计算辐射流（Radiance Flow）
        :param w2c1: 第一个相机位姿（世界到相机），形状为(4,4)或(batch,4,4)
        :param w2c2: 第二个相机位姿（世界到相机），形状同上
        :param gaussians: 高斯点数据，需包含位置、缩放、旋转等信息
                          格式示例: {'xyz': (N,3), 'scaling': (N,3), 'rotation': (N,4), ...}
        :param K: 相机内参矩阵，形状为(3,3)或(batch,3,3)
        :param image_size: 图像尺寸 (H, W)
        :return: 辐射流，形状为(1, 2, H, W)（与RAFT输出格式一致）
        """
        # 关键步骤（需根据你的高斯点渲染逻辑补充）：
        # 1. 从gaussians中获取3D位置 (xyz)
        # 2. 将3D点分别投影到两个相机的图像平面，得到像素坐标 (u1, v1) 和 (u2, v2)
        # 3. 计算光流：flow = (u2 - u1, v2 - v1)
        # 4. 生成密集光流图（需处理遮挡和空洞问题）

        # 临时返回零矩阵作为占位
        # TODO: 需替换为实际计算逻辑
        H, W = image_size
        radiance_flow = torch.zeros(1, 2, H, W, device=self.device)
        return radiance_flow

    def compute_prior_flow(self, 
                          img1: torch.Tensor, 
                          img2: torch.Tensor) -> torch.Tensor:
        """
        利用RAFT计算两个图像之间的先验光流（Prior Flow）
        :param img1: 第一张图像，形状为(1, 3, H, W)，值范围[0,1]
        :param img2: 第二张图像，形状同上
        :return: 先验光流，形状为(1, 2, H, W)（x和y方向的位移）
        """
        # 确保图像在正确设备上
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        # RAFT输入预处理（尺寸对齐）
        padder = InputPadder(img1.shape)
        img1_pad, img2_pad = padder.pad(img1, img2)

        # 计算光流（关闭梯度）
        with torch.no_grad():
            prior_flow = self.raft(img1_pad, img2_pad, iters=20, test_mode=True)[0]
        
        # 还原到原始尺寸
        prior_flow = padder.unpad(prior_flow)
        return prior_flow

    def compute_fds_loss(self, 
                        radiance_flow: torch.Tensor, 
                        prior_flow: torch.Tensor,
                        valid_mask: torch.Tensor = None) -> torch.Tensor:
        """
        计算FDS损失（辐射流与先验流的L2损失）
        :param radiance_flow: 辐射流，形状为(1, 2, H, W)
        :param prior_flow: 先验流，形状同上
        :param valid_mask: 有效区域掩码（1表示有效，0表示无效），形状为(1, 1, H, W)，可选
        :return: 标量损失值
        """
        # 确保流尺寸一致
        if radiance_flow.shape != prior_flow.shape:
            # 若尺寸不同，将辐射流上采样到先验流尺寸（RAFT输出可能下采样）
            radiance_flow = F.interpolate(
                radiance_flow, 
                size=prior_flow.shape[2:], 
                mode='bilinear', 
                align_corners=False
            )

        # 计算L2损失
        flow_diff = radiance_flow - prior_flow
        loss = torch.norm(flow_diff, p=2, dim=1).mean()

        # 应用有效掩码
        if valid_mask is not None:
            valid_mask = valid_mask.to(self.device)
            loss = (valid_mask * torch.norm(flow_diff, p=2, dim=1)).sum() / valid_mask.sum()

        return loss


# ------------------------------
# （测试用）
# ------------------------------
def test_fds_utils():
    # 1. 初始化RAFT模型（需替换为你的模型加载逻辑）
    from argparse import Namespace
    args = Namespace(
        small=False, 
        dropout=0, 
        alternate_corr=False, 
        mixed_precision=False
    )
    raft = RAFT(args).eval().cuda()
    raft.load_state_dict(torch.load("raft-things.pth"))  # 加载权重

    # 2. 创建FDS工具实例
    fds = FDSUtils(raft_model=raft)

    # 3. 测试相机扰动（临时返回原始位姿）
    w2c = torch.eye(4, device='cuda')  # 示例位姿
    perturbed_w2c = fds.generate_perturbed_pose(w2c)
    print("原始位姿与扰动位姿是否相同:", torch.allclose(w2c, perturbed_w2c))

    # 4. 测试光流和损失计算（使用随机数据）
    img1 = torch.rand(1, 3, 480, 640, device='cuda')  # 图像1
    img2 = torch.rand(1, 3, 480, 640, device='cuda')  # 图像2
    gaussians = {'xyz': torch.rand(1000, 3, device='cuda')}  # 示例高斯点
    K = torch.tensor([[500, 0, 320], [0, 500, 240], [0, 0, 1]], device='cuda')  # 示例内参

    # 计算光流
    prior_flow = fds.compute_prior_flow(img1, img2)
    radiance_flow = fds.compute_radiance_flow(w2c, perturbed_w2c, gaussians, K, (480, 640))
    
    # 计算损失
    fds_loss = fds.compute_fds_loss(radiance_flow, prior_flow)
    print(f"FDS损失值（随机数据）: {fds_loss.item()}")

if __name__ == "__main__":
    test_fds_utils()
