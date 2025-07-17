from .image_translation import ImageTranslation


# 注册自定义节点
NODE_CLASS_MAPPINGS = {
    "ImageTranslation": ImageTranslation
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTranslation": "Image Translation to BBox"
}
