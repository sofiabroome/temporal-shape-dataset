class ComposeMix(object):
    r"""Composes several transforms together. It takes a list of
    transformations, where each element odf transform is a list with 2
    elements. First being the transform function itself, second being a string
    indicating whether it's an "img" or "vid" transform
    Args:
        transforms (List[Transform, "<type>"]): list of transforms to compose.
                                                <type> = "img" | "vid"
    Example:
        >>> transforms.ComposeMix([
        [RandomCropVideo(84), "vid"],
        [torchvision.transforms.ToTensor(), "img"],
        [torchvision.transforms.Normalize(
                   mean=[0.485, 0.456, 0.406],  # default values for imagenet
                   std=[0.229, 0.224, 0.225]), "img"]
    ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, imgs):
        for t in self.transforms:
            if t[1] == "img":
                for idx, img in enumerate(imgs):
                    imgs[idx] = t[0](img)
            elif t[1] == "vid":
                imgs = t[0](imgs)
            else:
                print("Please specify the transform type")
                raise ValueError
        return imgs
