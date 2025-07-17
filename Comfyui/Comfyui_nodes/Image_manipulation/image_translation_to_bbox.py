import torch
import numpy as np


class ImageTranslation:
    """
    自定义 ComfyUI 节点：图像平移至掩膜的 bounding box。

    该节点将输入图像平移至指定掩膜的 bounding box 区域。它会根据 from_mask 的 bounding box 位置，将图像平移到 to_mask 的 bounding box 位置。
    如果 from_mask 的 bounding box 大于 to_mask 的 bounding box，则会缩放图像以适应新的 bounding box。
    如果任一掩膜没有有效区域，则返回原图像。

    输入:
    - image: 输入的图像张量，形状为 [Batch, Height, Width, Channels]。
    - from_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。
    - to_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。

    输出:
    - 处理后的图像张量，形状为 [Batch, Height, Width, Channels]。
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数。
        """
        return {
            "required": {
                "image": ("IMAGE",),  # 图像输入 (Tensor)
                "from_mask": ("MASK",),    # 从掩膜输入 (Tensor)
                "to_mask": ("MASK",), # 平移至掩膜 (Tensor)
            }
        }

    CATEGORY = "Custom/Image Processing"
    RETURN_TYPES = ("IMAGE",)  # 返回值类型为图像
    FUNCTION = "apply_mask"    # 节点主函数

    def apply_mask(self, image, from_mask, to_mask):
        """
        主逻辑：平移图片，从淹没 from_mask 的bounding box位置平移至 to_mask 的bounding box位置。

        参数:
        - image: 输入的图像张量，形状为 [Batch, Height, Width, Channels]。
        - from_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。
        - to_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。


        返回:
        - 处理后的图像张量，形状为 [Batch, Height, Width, Channels]。
        """
        # 确保图像和掩膜在同一个设备上（CPU/GPU）
        device = image.device
        from_mask = from_mask.to(device)
        to_mask = to_mask.to(device)

        # 调整图像维度为 [Batch, Channels, Height, Width]
        if len(from_mask.shape) == 4:  # 如果图像格式为 [Batch, Height, Width, Channels]
            from_mask = from_mask.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]
        if len(to_mask.shape) == 4:
            to_mask = to_mask.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]
        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]

        # 如果掩膜缺少通道维度，则添加通道维度
        if len(from_mask.shape) == 3:  # 掩膜格式为 [Batch, Height, Width]
            from_mask = from_mask.unsqueeze(1)  # 添加通道维度，变为 [Batch, 1, Height, Width]
        if len(to_mask.shape) == 3:  # 掩膜格式为 [Batch, Height, Width]
            to_mask = to_mask.unsqueeze(1)  # 添加通道维度，变为 [Batch, 1, Height, Width]

        # 分别计算 from_mask 和 to_mask 的bounding box top-left 和 bottom-right 坐标
        from_mask_bbox = torch.nonzero(from_mask, as_tuple=False)
        to_mask_bbox = torch.nonzero(to_mask, as_tuple=False)

        if from_mask_bbox.numel() == 0 or to_mask_bbox.numel() == 0:
            # 如果任一掩膜没有有效区域，则返回原图像
            return (image,)

        from_top_left = from_mask_bbox.min(dim=0)[0]
        from_bottom_right = from_mask_bbox.max(dim=0)[0]
        to_top_left = to_mask_bbox.min(dim=0)[0]
        to_bottom_right = to_mask_bbox.max(dim=0)[0]

        # 平移 from_mask bounding box 到 to_mask bounding box，使得二者中心对齐
        from_center = (from_top_left + from_bottom_right) / 2
        to_center = (to_top_left + to_bottom_right) / 2
        translation_vector = to_center - from_center
        
        # 创建平移矩阵
        translation_matrix = torch.tensor([[1, 0, translation_vector[0]],
                                           [0, 1, translation_vector[1]]], device=device)
        # 使用平移矩阵对图像进行平移
        grid = torch.nn.functional.affine_grid(translation_matrix.unsqueeze(0), image.size(), align_corners=False)
        image = torch.nn.functional.grid_sample(image.float(), grid, align_corners=False)

        # image 图像归一化到 [0, 1] 范围
        image = image.float() / 255.0

        # 图像先平移，后沿中心缩放
        # 判断 from_mask 是否能够放入 to_mask 的 bounding box
        if (to_bottom_right[2] - to_top_left[2] < from_bottom_right[2] - from_top_left[2]) or \
           (to_bottom_right[3] - to_top_left[3] < from_bottom_right[3] - from_top_left[3]):
            # 如果 from_mask 的 bounding box 大于 to_mask 的 bounding box，需要根据 from_mask 的bounding box 缩放比例
            scale_factor = min(
                (to_bottom_right[2] - to_top_left[2]) / (from_bottom_right[2] - from_top_left[2]),
                (to_bottom_right[3] - to_top_left[3]) / (from_bottom_right[3] - from_top_left[3])
            )

            # 对 image 按照 scale_factor 以图像中心进行缩放
            image = torch.nn.functional.interpolate(
                image, # 判断 image 图像格式
                scale_factor=scale_factor.item(),
                mode='bilinear',
                align_corners=False
            )

        # image 图像归一化回 [0, 255] 范围
        image = (image * 255.0).clamp(0, 255).to(torch.uint8)
            
        return (image,)


# 注册自定义节点
NODE_CLASS_MAPPINGS = {
    "ImageTranslation": ImageTranslation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTranslation": "Image Translation to BBox"
}



if __name__ == "__main__":
    from PIL import Image, ImageDraw
    
    from_mask = "../../test_imgs/ComfyUI_temp_bxabi_00001_.png"
    to_mask = "../../test_imgs/ComfyUI_01332_.png"
    image_path = "../../test_imgs/场景4_2.png"
    fill_device = "../../test_imgs/场景4.png"

    # display(Image.open(fill_device))
    image_pt = torch.from_numpy(np.array(Image.open(fill_device))).to("cpu")
    from_mask_pt = torch.from_numpy(np.array(Image.open(from_mask))).to("cpu")
    to_mask_pt = torch.from_numpy(np.array(Image.open(to_mask))).to("cpu")

    image_translation = ImageTranslation()

    # 格式转换
    image_pt = image_pt.unsqueeze(0)
    to_mask_pt = to_mask_pt.unsqueeze(0)
    from_mask_pt = from_mask_pt.unsqueeze(0)
    print(image_pt.shape, from_mask_pt.shape, to_mask_pt.shape)

    result = image_translation.apply_mask(image_pt, from_mask_pt, to_mask_pt)[0]