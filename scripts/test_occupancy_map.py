import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull

def rectangles_to_occupancy_map(all_obs, map_size):
    num_envs, num_rectangles, _ = all_obs.shape
    map_height, map_width = map_size

    # Create a tensor of zeros with the same size as the occupancy map
    occupancy_maps = torch.zeros((num_envs, num_rectangles, map_height, map_width), dtype=torch.float32)

    # Extract the center coordinates, quaternions, and sizes of all rectangles
    centers = all_obs[:, :, :2]
    quaternions = all_obs[:, :, 2:6]
    # print(quaternions)
    sizes = all_obs[:, :, 6:8]

    # Convert the quaternions to rotation matrices
    rotation_matrices = quaternion_to_matrix(quaternions)

    # print(rotation_matrices)

    # Convert the rectangles to sets of four points (one for each corner)
    points = torch.tensor([
        [-0.5, -0.5, 0],
        [0.5, -0.5, 0],
        [0.5, 0.5, 0],
        [-0.5, 0.5, 0]
    ])

    # widths, heights = sizes[:, :, 0], sizes[:, :, 1]

    # -widths/2, -heights/2, 0

    points = points.unsqueeze(0).unsqueeze(1).repeat(num_envs, num_rectangles, 1, 1)

    
    points[:, :, :, 0] *= sizes[:, :, 0].unsqueeze(-1)
    points[:, :, :, 1] *= sizes[:, :, 1].unsqueeze(-1)

    points[:, :, :, 0] *= map_width/5
    points[:, :, :, 1] *= map_height/2

    occupancy_maps[:, :, :, :] = 

    points = points @ rotation_matrices.transpose(2, 3)


    points[:, :, :, 0] += (centers[:, :, 0].unsqueeze(-1)* map_width/5)
    points[:, :, :, 1] += (centers[:, :, 1].unsqueeze(-1)*map_height/2)
    
    points = points.long()

    points[:, :, :, 0] = points[:, :, :, 0].clamp(min=0, max=map_width-1)
    points[:, :, :, 1] = points[:, :, :, 1].clamp(min=0, max=map_height-1)


    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            rect = points[i, j]

            
            
            min_x = min(rect[:, 0])
            max_x = max(rect[:, 0])
            min_y = min(rect[:, 1])
            max_y = max(rect[:, 1])


            for k in range(map_width):
                for l in range(map_height):
                    if (min_x <= k <= max_x and min_y <= l <= max_y):
                        occupancy_maps[i, l, k] = True

    return occupancy_maps

def quaternion_to_matrix(quaternions):
    """Converts a batch of quaternions to a batch of 3x3 rotation matrices"""
    num_envs, num_rectangles, _ = quaternions.shape
    w, x, y, z = quaternions[:, :, 0], quaternions[:, :, 1], quaternions[:, :, 2], quaternions[:, :, 3]
    xx = x * x
    xy = x * y
    xz = x * z
    xw = x * w
    yy = y * y
    yz = y * z
    yw = y * w
    zz = z * z
    zw = z * w

    print(yy, zz)

    return torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    ], dim=-1).view(num_envs, num_rectangles, 3, 3)

    # return torch.stack([
    #     1 - 2 * (yy + zz), 2 * (xy - zw), 2 * (xz + yw),
    #     2 * (xy + zw), 1 - 2 * (xx + zz), 2 * (yz - xw),
    #     2 * (xz - yw), 2 * (yz + xw), 1 - 2 * (xx + yy)
    # ], dim=-1)


if __name__ == '__main__':
    all_obs = torch.stack([torch.tensor([1.5, 1.0, 0.924, 0,   0, 0.383 , 2.0, 1.0]), torch.tensor([1.5, 1.0, 0.924, 0,   0, 0.383 , 2.0, 1.0])], dim=0)
    # all_obs = torch.tensor([[0.5, 1.0,    0.707, 0,   0.707,  0. , 2., 1.0]]).unsqueeze(0)
    all_obs1 = torch.stack([torch.tensor([1.5, 1.0, 0.7071068, 0, 0,  0.7071068 , 2, 1.0]), torch.tensor([1.5, 1.0, 1, 0, 0, 0 , 2, 1.0])], dim=0)
    all_obs = torch.stack([all_obs1, all_obs], dim=0)
    # print(all_obs.shape)
    
    occ_map = rectangles_to_occupancy_map(all_obs, (200, 500))
    print(occ_map.shape)

    import matplotlib.pyplot as plt

    plt.imshow(occ_map[0].squeeze(0).squeeze(0).numpy())
    plt.show()

    plt.imshow(occ_map[1].squeeze(0).squeeze(0).numpy())
    plt.show()


    

