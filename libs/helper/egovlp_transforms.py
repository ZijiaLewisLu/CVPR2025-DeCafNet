from torchvision import transforms
from torchvision.transforms._transforms_video import RandomCropVideo, RandomResizedCropVideo,CenterCropVideo, NormalizeVideo,ToTensorVideo,RandomHorizontalFlipVideo

def init_video_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    print('Video Transform is used!')
    normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            RandomResizedCropVideo(input_res, scale=randcrop_scale),
            RandomHorizontalFlipVideo(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict