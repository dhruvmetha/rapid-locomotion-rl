from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch


class AssetDef:
    def __init__(self, asset, name, base_position) -> None:
        self.asset = asset
        self.name = name
        self.base_position = base_position

class WorldAsset():
    def __init__(self, cfg, sim, gym) -> None:
        """
        Creates the world environment objects like obstacles, walls, etc.
        """
        self.envs = None # gets initialized in post_create_world
        self.gym = gym
        self.sim = sim
        self.cfg = cfg

        # initialize buffers and variables needed for the world actors
        self.handles = {}
        self.env_assets_map = {}
        self.env_actor_indices_map = {}
        self.all_actor_base_postions = {}
        

    def define_world(self):
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
            self.gym.create_box(self.sim, 3.5, .2, 1., asset_options), \
            self.gym.create_box(self.sim, 3.5, .2, 1., asset_options), \
            self.gym.create_box(self.sim, 0.2, 1.8, 1., asset_options), \
            self.gym.create_box(self.sim, 0.2, 1.8, 1., asset_options), \
                ]

        # their names
        # asset_names = ['ground', 'wall_left' , 'wall_right', 'wall_front, wall_back']
        asset_names = ['wall_left' , 'wall_right', 'wall_front', 'wall_back']

        # all base positions
        # asset_pos = [[0., 0., 0.005], [0., -2., .5],  [0., 2., .5], [2., 0., .5],  [-2., 0., .5]]
        asset_pos = [[0., -0.8, .5],  [0., 0.8, .5], [1.85, 0., .5],  [-1.85, 0., .5]]

        assets_container = [AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)]
            
        return assets_container

    def create_world(self, env_id, env_handle, env_origin):
        """
        environment setup, all the actors of the world get created here
        """
        
        if env_id not in self.handles:
            self.handles[env_id] = []
        
        assets_container = self.define_world()
        self.env_assets_map[env_id] = assets_container

        for asset in assets_container:
            pose = gymapi.Transform()
            pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
            pose.p = gymapi.Vec3(*pos)
            ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 1, 0)
            self.handles[env_id].append(ah)
        
    def post_create_world(self, envs, env_origins):
        """
        setup indices for resets of only these world actors
        """
        self.envs = envs
        self.env_origins = env_origins
     
    def init_buffers(self, **kwargs):
        """
        world buffers for actor root states, rb, contacts
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
                base_positions.append(asset.base_position)
                env_origins.append(self.env_origins[env_id])
        actor_indices = torch.tensor(actor_indices, dtype=torch.long, device='cuda:0')
        base_positions = torch.tensor(base_positions, dtype=torch.float32, device='cuda:0')
        env_origins = torch.vstack(env_origins)
        self.all_root_states[actor_indices, :3] = base_positions + env_origins
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_root_states), gymtorch.unwrap_tensor(actor_indices.to(dtype=torch.int32)), len(actor_indices))