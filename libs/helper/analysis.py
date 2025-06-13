
from .utils import compute_iou

def compute_iou_by_length(ckpt):

    dist_ious_pairs = []
    for vname, video in ckpt.videos.items():
        for i in range(len(video.abs_target)):
            ious = compute_iou(video.abs_target[i:i+1], video.results[i]['segments'].detach().cpu().numpy())
            if len(ious) == 0:
                continue
            ious = ious[0][-1]
            dist_ious_pairs.append([ ious, video.abs_target[i, 1] - video.abs_target[i, 0] ])
    dist_ious_pairs = np.array(dist_ious_pairs)

    ious = dist_ious_pairs[:, 0]
    (ious > 0.3).mean()

    import numpy as np

    # Convert dist_ious_pairs to a numpy array

    # Get the lengths from dist_ious_pairs
    lengths = dist_ious_pairs[:, 1]

    # Get the ious from dist_ious_pairs
    ious = dist_ious_pairs[:, 0]
    ious = ious > 0.3

    # Group the ious based on the lengths
    groups = np.histogram(lengths, bins=20)

    # Calculate the average iou in each group
    average_ious = [np.mean(ious[np.where((lengths >= groups[1][i]) & (lengths < groups[1][i+1]))]) for i in range(len(groups[1])-1)]

    # Print the average iou in each group
    # for i, avg_iou in enumerate(average_ious):
    #     print(f"Group {i+1}: Average IoU = {avg_iou}")
    
    return groups, average_ious

