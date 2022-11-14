from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch

from high_level_policy import *

class AssetDef:
    def __init__(self, asset, name, base_position) -> None:
        self.asset = asset
        self.name = name
        self.base_position = base_position

class WorldAsset():
    def __init__(self, cfg, sim, gym, device) -> None:
        """
        Creates the world environment objects like obstacles, walls, etc.
        """
        self.envs = None # gets initialized in post_create_world
        self.gym = gym
        self.sim = sim
        self.cfg = cfg
        self.num_envs = self.cfg.env.num_envs # gets initialized in post_create_world
        self.device = device

        # self.world_cfg = world_cfg()

        self.custom_box = world_cfg.CUSTOM_BLOCK
        # self.size_x_range = world_cfg.size_x_range
        # self.size_y_range = world_cfg.size_y_range

        # initialize buffers and variables needed for the world actors
        self.handles = {}
        self.env_assets_map = {}
        self.env_actor_indices_map = {}
        self.all_actor_base_postions = {}

        self.block_size = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float, requires_grad=False)
        self.block_weight = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float, requires_grad=False)

    def define_world(self, env_id):
        """
        define the world configuration and it's assets
        """
        # all assets 
        # gym_assets = [self.gym.create_box(self.sim, 6., 6., .01, gymapi.AssetOptions()), \
        #     self.gym.create_box(self.sim, 1., .2, 1., gymapi.AssetOptions()), \
        #     self.gym.create_box(self.sim, 1., .2, 1., gymapi.AssetOptions()), \
        #     self.gym.create_box(self.sim, 0.2, 1., 1., gymapi.AssetOptions()), \
        #     self.gym.create_box(self.sim, 0.2, 1., 1., gymapi.AssetOptions()), \
        #         ]

        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = True

        gym_assets = [ \
            self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
            self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
            self.gym.create_box(self.sim, 0.1, 1.8, 1., asset_options), \
                ]

        # their names
        # asset_names = ['ground', 'wall_left' , 'wall_right', 'wall_front, wall_back']
        asset_names = ['wall_left' , 'wall_right', 'wall_back']

        # all base positions
        # asset_pos = [[0., 0., 0.005], [0., -2., .5],  [0., 2., .5], [2., 0., .5],  [-2., 0., .5]]
        asset_pos = [[1., -1.0, .5],  [1., 1.0, .5], [-1.5, 0., .5]]

        # custom box
        if self.custom_box:
            # gym_assets.append(self.gym.create_box(self.sim, round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), 0.3, asset_options))
            # asset_names.append(world_cfg.fixed_block.name)
            # asset_pos.append([round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .15])

            asset_options = gymapi.AssetOptions()
            asset_options.disable_gravity = False
            asset_options.fix_base_link = True
            block_density = np.random.uniform(*world_cfg.movable_block.block_density_range)
            asset_options.density = block_density

            asset_size = [round(np.random.uniform(*world_cfg.movable_block.size_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]
            # if np.random.uniform(0, 1) < 0.1:
            #     asset_size[1] = round(np.random.uniform(*[0.8, 1.3]), 2)

            self.block_size[env_id, :] = torch.tensor(asset_size[:2])
            self.block_weight[env_id, 0] = (block_density * np.prod(asset_size))
            gym_assets.append(self.gym.create_box(self.sim, asset_size[0], asset_size[1], asset_size[2], asset_options))
            asset_names.append(world_cfg.movable_block.name)
            asset_pos.append([round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2])

        assets_container = [AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)]

        return assets_container

    def create_world(self, env_id, env_handle, env_origin):
        """
        environment setup, all the actors of the world get created here
        """
        
        if env_id not in self.handles:
            self.handles[env_id] = []
        
        assets_container = self.define_world(env_id)
        self.env_assets_map[env_id] = assets_container

        for asset in assets_container:
            pose = gymapi.Transform()
            pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
            pose.p = gymapi.Vec3(*pos)
            ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 0, 0)
            self.handles[env_id].append(ah)
        
    def post_create_world(self, envs, env_origins):
        """
        setup indices for resets of only these world actors
        """
        self.envs = envs
        self.env_origins = env_origins
     
    def init_buffers(self, **kwargs):
        """
        world buffers for actor root states, rb, coselntacts
        """
        self.all_root_states = kwargs['root_states']
        self.all_dof_state = kwargs['dof_states']
        self.all_rigid_body_state = kwargs['rigid_body_states']
        self.all_contact_forces = kwargs['contact_forces']
        return
        
    def reset_world(self, env_ids, _):
        """
        reset the world actors in the environment
        """
        # TODO: precompute these lists for more efficiency in post_create_world
        actor_indices = []
        base_positions = []
        env_origins = []
        for env_id in env_ids:
            env_id = env_id.item()
            env_handle = self.envs[env_id]
            assets_container = self.env_assets_map[env_id]
            for asset in assets_container:
                actor_indices.append(self.gym.find_actor_index(env_handle, asset.name, gymapi.DOMAIN_SIM))
                env_origins.append(self.env_origins[env_id])
                if asset.name == world_cfg.movable_block.name:
                    base_positions.append([round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2])

                elif asset.name == world_cfg.fixed_block.name:
                    base_positions.append([round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2])
                # elif asset.name == 'wall_front':
                #     if np.random.uniform(0, 1) > 0.5:
                #         base_positions.append([3.2, -0.35, .5])
                #     else:
                #         base_positions.append([3.2, 0.35, .5])
                    # print(asset.name, asset.base_position, base_positions[-1])
                else:
                    base_positions.append(asset.base_position)
        actor_indices = torch.tensor(actor_indices, dtype=torch.long, device='cuda:0')
        base_positions = torch.tensor(base_positions, dtype=torch.float32, device='cuda:0')
        env_origins = torch.vstack(env_origins)
        self.all_root_states[actor_indices, :3] = base_positions + env_origins
        self.all_root_states[actor_indices, 3:] = 0.
        self.all_root_states[actor_indices, 6] = 1. 
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_root_states), gymtorch.unwrap_tensor(actor_indices.to(dtype=torch.int32)), len(actor_indices))

    def get_block_obs(self):
        if self.custom_box:
            block_ids_int32 = torch.tensor([self.gym.find_actor_index(self.envs[i], world_cfg.movable_block.name, gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)
            obs = torch.cat([(self.all_root_states[block_ids_int32, :2] - self.env_origins[:, :2]).clone(), self.all_root_states[block_ids_int32, 3:7].clone(), self.block_size.clone(), self.block_weight.clone()], dim=-1)

            if world_cfg.fixed_block.add_to_obs:
                fixed_block_ids = torch.tensor([self.gym.find_actor_index(self.envs[i], world_cfg.fixed_block.name, gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)
                obs = torch.cat([obs, (self.all_root_states[fixed_block_ids, :2] - self.env_origins[:, :2]).clone()], dim=-1)
            return obs
        if world_cfg.fixed_block.add_to_obs:
            return torch.zeros(self.num_envs, 11, device=self.device)
        
        return torch.zeros(self.num_envs, 9, device=self.device)
