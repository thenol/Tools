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
                "from_bbox": ("BBOX",),  
                "to_bbox": ("BBOX",),    
            },
            "optional": {
                "from_mask": ("MASK",),    # 从掩膜输入 (Tensor)
                "to_mask": ("MASK",), # 平移至掩膜 (Tensor)
                "scale_default": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.001,
                    "round": 0.0001, 
                    "display": "number"})
            }
        }

    CATEGORY = "Custom/Image Processing"
    RETURN_TYPES = ("IMAGE",)  # 返回值类型为图像
    FUNCTION = "apply_mask"    # 节点主函数

    def apply_mask(self, image, from_bbox=None, to_bbox=None, scale_default=0.997, from_mask=None, to_mask=None):
        """
        主逻辑：平移图片，从淹没 from_mask 的bounding box位置平移至 to_mask 的bounding box位置。

        参数:
        - image: 输入的图像归一化之后的张量，形状为 [Batch, Height, Width, Channels]。
        - from_bbox: 输入的 bounding box，格式为 [x, y, width, height]，其中 (x, y) 是左上角坐标，width 和 height 分别是宽度和高度。
        - to_bbox: 输入的 bounding box，格式为 [x, y, width, height]，其中 (x, y) 是左上角坐标，width 和 height 分别是宽度和高度。
        - from_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。
        - to_mask: 输入的掩膜张量，形状为 [Batch, Height, Width] 或 [Batch, 1, Height, Width]。


        返回:
        - 处理后的图像归一化后的张量，形状为 [Batch, Height, Width, Channels]。
        """
        assert (from_bbox is not None and to_bbox is not None) or (from_mask is not None and to_mask is not None), \
            "Either from_bbox and to_bbox or from_mask and to_mask must be provided."

        device = image.device
        # 如果提供了 bounding box，则将其转换为掩膜
        if from_bbox is not None and to_bbox is not None:
            from_mask = torch.zeros_like(image, dtype=torch.uint8, device=device)
            to_mask = torch.zeros_like(image, dtype=torch.uint8, device=device)

            # 将 from_bbox 和 to_bbox 转换为掩膜
            from_x, from_y, from_w, from_h = from_bbox
            to_x, to_y, to_w, to_h = to_bbox
            from_mask[:, from_y:from_y + from_h, from_x:from_x + from_w] = 255
            to_mask[:, to_y:to_y + to_h, to_x:to_x + to_w] = 255

        # 确保图像和掩膜在同一个设备上（CPU/GPU）
        from_mask = from_mask.to(device)
        to_mask = to_mask.to(device)

        # 调整图像维度为 [Batch, Channels, Height, Width]
        if len(from_mask.shape) == 4:  # 如果图像格式为 [Batch, Height, Width, Channels]
            from_mask = from_mask.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]
        if len(to_mask.shape) == 4:
            to_mask = to_mask.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]

        # 如果掩膜缺少通道维度，则添加通道维度
        if len(from_mask.shape) == 3:  # 掩膜格式为 [Batch, Height, Width]
            from_mask = from_mask.unsqueeze(1)  # 添加通道维度，变为 [Batch, 1, Height, Width]
        if len(to_mask.shape) == 3:  # 掩膜格式为 [Batch, Height, Width]
            to_mask = to_mask.unsqueeze(1)  # 添加通道维度，变为 [Batch, 1, Height, Width]

        # 分别计算 from_mask 和 to_mask 的bounding box top-left 和 bottom-right 坐标
        from_mask_bbox = torch.nonzero(from_mask, as_tuple=False)
        to_mask_bbox = torch.nonzero(to_mask, as_tuple=False)

        if from_mask_bbox.numel() == 0 or to_mask_bbox.numel() == 0:
            return (image,)

        from_top_left = from_mask_bbox.min(dim=0)[0]
        from_bottom_right = from_mask_bbox.max(dim=0)[0]
        to_top_left = to_mask_bbox.min(dim=0)[0]
        to_bottom_right = to_mask_bbox.max(dim=0)[0]

        if len(image.shape) == 4:
            image = image.permute(0, 3, 1, 2)  # 转换为 [Batch, Channels, Height, Width]

        obj_image_region = image[:, :, from_top_left[2]:from_bottom_right[2], from_top_left[3]:from_bottom_right[3]]
        transparent_bg = torch.zeros_like(image, device=device)  # 创建透明背景

        # 图像先平移，后沿中心缩放
        # 判断 from_mask 是否能够放入 to_mask 的 bounding box
        scale_factor = torch.tensor(1.0, device=device)
        if (to_bottom_right[2] - to_top_left[2] < from_bottom_right[2] - from_top_left[2]) or \
           (to_bottom_right[3] - to_top_left[3] < from_bottom_right[3] - from_top_left[3]):
            # 如果 from_mask 的 bounding box 大于 to_mask 的 bounding box，需要根据 from_mask 的bounding box 缩放比例
            scale_factor = min(
                (to_bottom_right[2] - to_top_left[2]) / (from_bottom_right[2] - from_top_left[2]),
                (to_bottom_right[3] - to_top_left[3]) / (from_bottom_right[3] - from_top_left[3])
            )

            # 将 image 中 from_bbox 区域 按照 scale_factor 以图像中心为中心进行缩放
            if scale_default is not None and scale_default != 0:
                s_factor = scale_default
            else:
                s_factor = scale_factor.item()
            obj_image_region = torch.nn.functional.interpolate(
                obj_image_region, # 判断 image 图像格式
                scale_factor=s_factor,
                mode='bilinear',
                align_corners=False
            )
        
        # 计算 to_mask 的中心点
        from_center = (from_top_left + from_bottom_right) / 2
        to_center = (to_top_left + to_bottom_right) / 2
        obj_b, obj_c, obj_h, obj_w = obj_image_region.shape

        # 计算 obj_image_region 在 to_mask 中的放置位置
        place_y = int(to_center[2] - obj_h / 2)
        place_x = int(to_center[3] - obj_w / 2)

        print(f"original image bbox:{from_bbox}, from_box center (w, h) is {from_center[2:].int().tolist()[::-1]}")
        print(f"reference image bbox:{to_bbox}, to_box center (w, h) is {to_center[2:].int().tolist()[::-1]}")
        print(f"translated image bbox:{[place_x, place_y, obj_h, obj_w]}, translated image center (w, h) is {[place_x + obj_w / 2, place_y + obj_h / 2]}")
        print(f"scale_factor is {scale_factor.item()}, default factor is {scale_default},  use factor {s_factor}")

        transparent_bg[:, :, place_y:place_y + obj_h, place_x:place_x + obj_w] = obj_image_region
        image = transparent_bg

        image = image.permute(0, 2, 3, 1)  # 转换为 [Batch, Height, Width, Channels]

        # 返回处理后的图像
        return (image,)


if __name__ == "__main__":
    from PIL import Image, ImageDraw
    
    # from_mask = "../../test_imgs/ComfyUI_temp_bxabi_00001_.png"
    # to_mask = "../../test_imgs/ComfyUI_01332_.png"
    image_path = "../../test_imgs/背景C.jpg"
    fill_device = "../../test_imgs/C9.png"

    # display(Image.open(fill_device))
    image_pt = torch.from_numpy(np.array(Image.open(fill_device))).to("cpu")
    # from_mask_pt = torch.from_numpy(np.array(Image.open(from_mask))).to("cpu")
    # to_mask_pt = torch.from_numpy(np.array(Image.open(to_mask))).to("cpu")

    image_translation = ImageTranslation()

    # 格式转换
    image_pt = image_pt.unsqueeze(0)
    # to_mask_pt = to_mask_pt.unsqueeze(0)
    # from_mask_pt = from_mask_pt.unsqueeze(0)
    # print(image_pt.shape, from_mask_pt.shape, to_mask_pt.shape)

    result = image_translation.apply_mask(image_pt, from_bbox=[771, 759, 405, 567], to_bbox=[773, 760, 401, 556], from_mask=None, to_mask=None)[0]
