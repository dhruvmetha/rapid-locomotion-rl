from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch
import time

from high_level_policy import *

class AssetDef:
    def __init__(self, asset, name, base_position) -> None:
        self.asset = asset
        self.name = name
        self.base_position = base_position

class WorldSetup:
    def __init__(self):
        pass
    def define_world(self):
        pass
    def reset_world(self):
        pass

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
        self.variables = None

        self.contact_memory_time = 20

        # self.block_size = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float, requires_grad=False)
        # self.block_weight = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float, requires_grad=False)

        # self.fixed_block_size = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float, requires_grad=False)

        # self.block_contact_buf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        # self.block_contact_ctr = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.inplay_assets = [
        {
            'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'],
            'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
            'pos': [[1.5, 0.5, 0.25], [2.25, -0.15, 0.25], [1.5, -0.4, 0.25]],
            'density': [10000, 10000, 3]
        },
        {
            'name': ['fixed_asset_3', 'fixed_asset_4', 'movable_asset_2'],
            'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
            'pos': [[1.5, 0.5, 0.25], [2.25, -0.6, 0.25], [1.5, -0.4, 0.25]],
            'density': [10000, 10000, 3]
        },
        
        # {
        #     'name': ['fixed_asset_5',  'movable_asset_3'],
        #     'size': [[0.5, 0.8, 0.5], [0.25, 0.9, 0.5]],
        #     'pos': [[1.5, 0.5, 0.25], [1.3, -0.4, 0.25]],
        #     'density': [10000, 3]
        # },

        {
            'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'],
            'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
            'pos': [[1.5, -0.5, 0.25], [2.25, 0.15, 0.25], [1.5, 0.4, 0.25]],
            'density': [10000, 10000, 3]
        },
        {
            'name': ['fixed_asset_8', 'fixed_asset_9', 'movable_asset_5'],
            'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
            'pos': [[1.5, -0.5, 0.25], [2.25, 0.6, 0.25], [1.5, 0.4, 0.25]],
            'density': [10000, 10000, 3]
        },
        
        # {
        #     'name': ['fixed_asset_10',  'movable_asset_6'],
        #     'size': [[0.5, 0.8, 0.5], [0.25, 0.9, 0.5]],
        #     'pos': [[1.5, -0.5, 0.25], [1.3, 0.4, 0.25]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'],
        #     'size': [[0.5, 0.8, 0.5], [0.5, 0.6, 0.3], [0.25, 0.9, 0.5]],
        #     'pos': [[0.8, 0.5, 0.25], [2., -0.4, 0.25], [1.5, -0.4, 0.25]],
        #     'density': [10000, 10000, 5]
        # },
        {
            'name': ['fixed_block', 'movable_block'],
            'size': [[round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), 0.3], [round(np.random.uniform(*world_cfg.movable_block.size_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .15], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]],
            'density': [10000, 3]
        },
        # {
        #     'name': ['movable_block_1'],
        #     'size': [[round(np.random.uniform(*world_cfg.movable_block.size_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]],
        #     'pos': [[round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]],
        #     'density': [3]
        # },
        # {
        #     'name': ['fixed_block_test'],
        #     'size': [[round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), 0.3]],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .15]],
        #     'density': [10000]
        # }
        ]

        self.inplay = {}

        self.base_assets = []
        self.asset_counts = 0
        self.asset_idx_map = {}
        self.idx_asset_map = {}
        self.env_assetname_map = {i: {} for i in range(self.num_envs)}
        self.env_assetname_bool_map = {i: {} for i in range(self.num_envs)}
        idx_ctr = 0
        for i in self.inplay_assets:
            self.asset_counts += len(i['name'])
            for j in i['name']:
                self.asset_idx_map[j] = idx_ctr
                self.idx_asset_map[idx_ctr] = j
                idx_ctr += 1
                for k in range(self.num_envs):
                    self.env_assetname_map[k][j] = []
                    self.env_assetname_bool_map[k][j] = False
        

        self.block_size = torch.zeros((self.num_envs, self.asset_counts, 2), device=self.device, dtype=torch.float, requires_grad=False)
        self.block_weight = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.float, requires_grad=False)

        self.block_contact_buf = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.block_contact_ctr = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.fixed_block_contact_buf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.fixed_block_contact_ctr = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.obs = torch.zeros((self.num_envs, 24), device=self.device, dtype=torch.bool, requires_grad=False)


        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = False
        asset_options.fix_base_link = True

        gym_assets = [ \
            self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
            self.gym.create_box(self.sim, 9.0, .1, 1., asset_options), \
            self.gym.create_box(self.sim, 0.1, 1.8, 1., asset_options), \
                ]
        asset_names = ['wall_left' , 'wall_right', 'wall_back']

        # all base positions
        asset_pos = [[1., -1.0, .5],  [1., 1.0, .5], [-1.5, 0., .5]]

        self.base_assets = [AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)]

        self.env_asset_ctr = torch.arange(0, 8, dtype=torch.long, device=self.device).repeat(self.num_envs, 1)
        self.env_asset_bool = torch.zeros(self.num_envs, self.asset_counts, dtype=torch.bool, device=self.device)
        self.all_env_ids = torch.arange(0, self.num_envs, 1, dtype=torch.long, device=self.device).view(-1, 1)

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

        

        

        # self.inplay_assets = {
        #     'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'],
        #     'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
        #     'pos': [[1.5, 0.5, 0.25], [2.0, -0.2, 0.25], [1.5, -0.4, 0.25]],
        #     'density': [1000, 1000, 5]
        # }

        # self.inplay_assets = {
        #     'name': ['fixed_block', 'movable_block'],
        #     'size': [[round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), 0.3], [round(np.random.uniform(*world_cfg.movable_block.size_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .15], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]],
        #     'density': [1000, 5]
        # }
        assets_container = []

        for i in self.inplay_assets:
            gym_assets = []
            asset_names = []
            asset_pos = []

            for idx, name in enumerate(i['name']):
                asset_options = gymapi.AssetOptions()
                asset_options.disable_gravity = False
                asset_options.fix_base_link = False
                asset_options.density = i['density'][idx]

                gym_assets.append(self.gym.create_box(self.sim, *(i['size'][idx]), asset_options))
                
                self.block_size[env_id, self.asset_idx_map[name], :] = torch.tensor(i['size'][idx][:2])
                
                asset_names.append(i['name'][idx])
                asset_pos.append(i['pos'][idx])
            assets_container.append([AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)])
        

        # for idx in range(len(self.inplay_assets['name'])):
        #     asset_options = gymapi.AssetOptions()
        #     asset_options.disable_gravity = False
        #     asset_options.fix_base_link = False
        #     asset_options.density = self.inplay_assets['density'][idx]
            
        #     gym_assets.append(self.gym.create_box(self.sim, *(self.inplay_assets['size'][idx]), asset_options))
        #     self.block_size[env_id, idx, :] = torch.tensor(self.inplay_assets['size'][idx][:2])
        #     asset_names.append(self.inplay_assets['name'][idx])
        #     asset_pos.append(self.inplay_assets['pos'][idx])


        # custom box
        # if self.custom_box:
        #     asset_options = gymapi.AssetOptions()
        #     asset_options.disable_gravity = False
        #     asset_options.fix_base_link = False
        #     asset_options.density = 10000
        #     asset_size = [round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), 0.3]
        #     gym_assets.append(self.gym.create_box(self.sim, *asset_size, asset_options))
        #     asset_names.append(world_cfg.fixed_block.name)
        #     asset_pos.append([round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .15])
        #     self.fixed_block_size[env_id, :] = torch.tensor(asset_size[:2])
            
        #     asset_options = gymapi.AssetOptions()
        #     asset_options.disable_gravity = False
        #     asset_options.fix_base_link = False
        #     block_density = np.random.uniform(*world_cfg.movable_block.block_density_range)
        #     asset_options.density = block_density

        #     asset_size = [round(np.random.uniform(*world_cfg.movable_block.size_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]
        #     # if np.random.uniform(0, 1) < 0.1:
        #     #     asset_size[1] = round(np.random.uniform(*[0.8, 1.3]), 2)

        #     self.block_size[env_id, :] = torch.tensor(asset_size[:2])
        #     self.block_weight[env_id, 0] = (block_density * np.prod(asset_size))
        #     gym_assets.append(self.gym.create_box(self.sim, asset_size[0], asset_size[1], asset_size[2], asset_options))
        #     asset_names.append(world_cfg.movable_block.name)
        #     asset_pos.append([round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2])
        return assets_container

    def add_variables(self, **kwargs):
        print('here', kwargs)
        self.variables = kwargs

    def create_world(self, env_id, env_handle, env_origin):
        """
        environment setup, all the actors of the world get created here
        """

        
        
        if env_id not in self.handles:
            self.handles[env_id] = []

        for asset in self.base_assets:
            pose = gymapi.Transform()
            pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
            pose.p = gymapi.Vec3(*pos)
            ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 0, 0)
            # self.handles[env_id].append(ah)

        assets_container = self.define_world(env_id)
        self.env_assets_map[env_id] = assets_container

        for asset_container in assets_container:
            for asset in asset_container:
                pose = gymapi.Transform()
                pos = env_origin.clone(); pos[0] += asset.base_position[0]; pos[1] += asset.base_position[1]; pos[2] += asset.base_position[2]
                pose.p = gymapi.Vec3(*pos)
                ah = self.gym.create_actor(env_handle, asset.asset, pose, asset.name, env_id, 0, 0)
                self.handles[env_id].append(ah)

        self.asset_root_ids = {}
        self.asset_contact_ids = {}

        

        
    def post_create_world(self, envs, env_origins):
        """
        setup indices for resets of only these world actors
        """
        self.envs = envs
        self.env_origins = env_origins

        for name in self.asset_idx_map.keys():
            self.asset_root_ids[name] = torch.tensor([self.gym.find_actor_index(self.envs[i], name, gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)
            block_actor_handles = [self.gym.find_actor_handle(self.envs[i], name) for i in range(self.num_envs)]            
            self.asset_contact_ids[name] = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs, block_actor_handles)], dtype=torch.long, device=self.device)
     
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
        env_asset_bool = []
        self.env_asset_bool[env_ids, :] = False

        for env_id in env_ids:
            
            env_id = env_id.item()
            env_handle = self.envs[env_id]
            assets_container = self.env_assets_map[env_id]
            random_idx = np.random.choice(np.arange(0, len(assets_container)))
            asset_bool = [False]*self.asset_counts
            # print(random_idx)
            self.inplay[env_id] = random_idx
            away_indices = []
            in_indices = []
            for idx, asset_container in enumerate(assets_container):
                for asset in asset_container:
                    actor_indices.append(self.gym.find_actor_index(env_handle, asset.name, gymapi.DOMAIN_SIM))
                    env_origins.append(self.env_origins[env_id])
                    if idx == random_idx:
                        # asset_bool[self.asset_idx_map[asset.name]] = True
                        in_indices.append((actor_indices[-1], asset.name, env_id))

                        self.env_asset_bool[env_id, self.asset_idx_map[asset.name]] = True
                        if asset.name == world_cfg.movable_block.name:
                            base_positions.append([round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2])

                        elif asset.name == world_cfg.fixed_block.name or asset.name.startswith('fixed_block'):
                            # if np.random.uniform(0, 1) < 0.6:
                            #     base_positions.append([round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(-0.4, 0.4), 2), 0.2])
                            # else:
                            base_positions.append([round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2])
                        # elif asset.name == 'wall_front':
                        #     if np.random.uniform(0, 1) > 0.5:
                        #         base_positions.append([3.2, -0.35, .5])
                        #     else:
                        #         base_positions.append([3.2, 0.35, .5])
                            # print(asset.name, asset.base_position, base_positions[-1])
                        else:
                            base_positions.append(asset.base_position)
                    else:
                        # asset_bool[self.asset_idx_map[asset.name]] = False
                        self.env_asset_bool[env_id, self.asset_idx_map[asset.name]] = False
                        base_positions.append([bp - 100000.0 for bp in asset.base_position])
                        away_indices.append((actor_indices[-1], env_id, self.asset_idx_map[asset.name]))

            # env_asset_bool.append(asset_bool)
                    
        # self.env_asset_bool[env_ids, :] = torch.tensor(env_asset_bool, dtype=torch.bool, device=self.device)
        # print('here', self.env_asset_bool[0])
        # self.env_asset_ctr[env_ids, :] = torch.arange(0, 8, dtype=torch.long, device=self.device)

        
        
        actor_indices = torch.tensor(actor_indices, dtype=torch.long, device='cuda:0')
        in_indices_t = torch.tensor([i for i, _, _ in in_indices], dtype=torch.long, device='cuda:0')
        in_env_ids = torch.tensor([i for _, _, i in in_indices], dtype=torch.long, device='cuda:0')
        in_asset_names = [i for _, i, _ in in_indices]
        base_positions = torch.tensor(base_positions, dtype=torch.float32, device='cuda:0')
        
        env_origins = torch.vstack(env_origins)
        self.all_root_states[actor_indices, :3] = base_positions + env_origins
        self.all_root_states[actor_indices, 3:] = 0.
        self.all_root_states[actor_indices, 6] = 1. 
        self.block_contact_buf[env_ids, :, :] = False
        self.block_contact_ctr[env_ids, :, :] = self.contact_memory_time + 1

        for aw_id, aw_env, aw_name in zip(in_indices_t, in_env_ids, in_asset_names):
            # print(self.all_root_states[aw_id])
            if (torch.linalg.norm(self.all_root_states[aw_id, :2]).item() > 1000):
                print("##########")
                print(aw_name, aw_env, self.env_asset_bool[aw_env][aw_name])
                print("##########")
        


        # self.fixed_block_contact_buf[:] = False
        # self.fixed_block_contact_ctr[:] = self.contact_memory_time + 1

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_root_states), gymtorch.unwrap_tensor(actor_indices.to(dtype=torch.int32)), len(actor_indices))

    def get_block_obs(self):

        # start = time.time()
        ids = []
        block_actor_handles = []
        ids_contact = []
        all_obs = torch.zeros((self.num_envs, 8*4), dtype=torch.float, device=self.device)
        self.env_asset_ctr[:, :] = torch.arange(0, 8, dtype=torch.long, device=self.device)

        for _, name in enumerate(self.asset_idx_map.keys()):
            ids = self.asset_root_ids[name]
            # print(name, ids)
            ids_contact = self.asset_contact_ids[name]


            self.env_asset_bool[:, self.asset_idx_map[name]] = ~(self.all_root_states[ids, 0] < -100)
            
            all_obs[self.all_env_ids, self.env_asset_ctr] += (torch.cat([(self.all_root_states[ids, :2] - self.env_origins[:, :2]).clone(), self.all_root_states[ids, 3:7].clone(), self.block_size[:, self.asset_idx_map[name], :].clone()], dim=-1) * self.env_asset_bool[:, self.asset_idx_map[name]].view(-1, 1))


            # all_obs[all_obs[self.all_env_ids, self.env_asset_ctr] < -100] = 0.

            # tester = torch.linalg.norm(all_obs[self.all_env_ids, self.env_asset_ctr[:, :2]], dim=-1)
            # tester_mask = tester!=0
            # tester_mean = (tester*tester_mask).sum(dim=0)/tester_mask.sum(dim=0)
            # if torch.abs(tester_mean) > 100:
            #     try:
            #         print("###########################################################################")
            #         fake_env_ids = self.all_env_ids[torch.linalg.norm(all_obs[self.all_env_ids, self.env_asset_ctr[:, :2]], dim=-1) > 100]
            #         print(name, self.env_asset_bool[fake_env_ids])
            #         print(fake_env_ids, all_obs[fake_env_ids[0], :2])
            #         print("###########################################################################")
            #     except:
            #         pass

            # envss = ((all_obs[self.all_env_ids, self.env_asset_ctr[0]] < -10)[:2]).nonzero(as_tuple=True)[0]
            # # print(envss)
            # if len(envss) > 0:
            #     print(self.env_asset_bool[envss, self.asset_idx_map[name]])

            block_contact_buf = (torch.linalg.norm(self.all_contact_forces[ids_contact, :2], dim=-1) > 1.).view(-1, 1)
            self.block_contact_ctr[:, self.asset_idx_map[name], :] = (~block_contact_buf) * self.block_contact_ctr[:, self.asset_idx_map[name], :]
            self.block_contact_buf[:, self.asset_idx_map[name], :] = self.block_contact_ctr[:, self.asset_idx_map[name], :] < self.contact_memory_time
            self.block_contact_ctr[:, self.asset_idx_map[name], :][self.block_contact_ctr[:, self.asset_idx_map[name]] < self.contact_memory_time] += 1

            all_obs[self.all_env_ids, self.env_asset_ctr] *= self.block_contact_buf[:, self.asset_idx_map[name], :]
            self.env_asset_ctr += 8*(self.env_asset_bool[:, self.asset_idx_map[name]].view(-1, 1).int())
        
        return all_obs

        for env_id in range(self.num_envs):
            inplay_assets = self.env_assets_map[env_id][self.inplay[env_id]]
            # print(self.env_assets_map[env_id], self.inplay[env_id])
            # print(env_id)
            for idx, asset in enumerate(inplay_assets):
                # print(env_id, asset.name)
                obs_idx = idx * 8
                # if self.env_as
                if not self.env_assetname_bool_map[env_id][asset.name]:
                    self.env_assetname_map[env_id][asset.name].append(self.gym.find_actor_index(self.envs[env_id], asset.name, gymapi.DOMAIN_SIM))
                    self.env_assetname_map[env_id][asset.name].append(self.gym.find_actor_handle(self.envs[env_id], asset.name))
                    self.env_assetname_map[env_id][asset.name].append(self.gym.find_actor_rigid_body_index(self.envs[env_id], self.env_assetname_map[env_id][asset.name][-1], "box", gymapi.DOMAIN_SIM))
                
                    self.env_assetname_bool_map[env_id][asset.name] = True
                
                ids = self.env_assetname_map[env_id][asset.name][0]
                block_actor_handle = self.env_assetname_map[env_id][asset.name][1]
                id_contact = self.env_assetname_map[env_id][asset.name][2]
                # ids = self.gym.find_actor_index(self.envs[env_id], asset.name, gymapi.DOMAIN_SIM)
                # print(torch.cat([(self.all_root_states[ids, :2] - self.env_origins[env_id, :2]).clone(), self.all_root_states[ids, 3:7].clone(), self.block_size[env_id, self.asset_idx_map[asset.name], :].clone()], dim=0).view(1, -1).shape)
                # print(all_obs[env_id, obs_idx: obs_idx+8])
                all_obs[env_id, obs_idx: obs_idx+8] = torch.cat([(self.all_root_states[ids, :2] - self.env_origins[env_id, :2]).clone(), self.all_root_states[ids, 3:7].clone(), self.block_size[env_id, self.asset_idx_map[asset.name], :].clone()], dim=0).view(1, -1)

                # block_actor_handle = self.gym.find_actor_handle(self.envs[env_id], asset.name)
                # id_contact = self.gym.find_actor_rigid_body_index(self.envs[env_id], block_actor_handle, "box", gymapi.DOMAIN_SIM)
                
                block_contact_buf = (torch.linalg.norm(self.all_contact_forces[id_contact, :2], dim=-1) > 1.).view(-1, 1)
                self.block_contact_ctr[env_id, self.asset_idx_map[asset.name], :] = ~(block_contact_buf) * self.block_contact_ctr[env_id, self.asset_idx_map[asset.name], :]
                self.block_contact_buf[env_id, self.asset_idx_map[asset.name], :] = self.block_contact_ctr[env_id, self.asset_idx_map[asset.name], :] < self.contact_memory_time

                self.block_contact_ctr[env_id, self.asset_idx_map[asset.name], :][self.block_contact_ctr[env_id, self.asset_idx_map[asset.name]] < self.contact_memory_time] += 1

                all_obs[env_id, obs_idx: obs_idx+8] *= self.block_contact_buf[env_id, self.asset_idx_map[asset.name], :]
        print(time.time()-start)

        return all_obs

        ids = torch.tensor(ids, dtype=torch.long, device=self.device)
        ids_contact = torch.tensor(ids_contact, dtype=torch.long, device=self.device)

        obs = torch.cat([(self.all_root_states[ids, :2] - self.env_origins[:, :2]).clone(), self.all_root_states[ids, 3:7].clone(), self.block_size[:, idx, :].clone()], dim=-1)

        all_obs = None
        for idx in range(len(self.inplay_assets['name'])):
            
            ids = torch.tensor([self.gym.find_actor_index(self.envs[i], self.inplay_assets['name'][idx], gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)
            block_actor_handles = [self.gym.find_actor_handle(self.envs[i], self.inplay_assets['name'][idx]) for i in range(self.num_envs)]            
            ids_contact = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs, block_actor_handles)], dtype=torch.long, device=self.device)
            
            obs = torch.cat([(self.all_root_states[ids, :2] - self.env_origins[:, :2]).clone(), self.all_root_states[ids, 3:7].clone(), self.block_size[:, idx, :].clone()], dim=-1)

            block_contact_buf = (torch.linalg.norm(self.all_contact_forces[ids_contact, :2], dim=-1) > 1.).view(-1, 1)
            self.block_contact_ctr[:, idx, :] = (~block_contact_buf) * self.block_contact_ctr[:, idx, :]
            self.block_contact_buf[:, idx, :] = self.block_contact_ctr[:, idx, :] < self.contact_memory_time
            # print(self.block_contact_ctr[(self.block_contact_ctr[:, idx] < self.contact_memory_time).repeat(2, 2, 1)])
            self.block_contact_ctr[:, idx, :][self.block_contact_ctr[:, idx] < self.contact_memory_time] += 1

            obs *= self.block_contact_buf[:, idx].view(-1, 1)

            if idx == 0:
                all_obs = obs
            else:
                all_obs = torch.cat([all_obs, obs], dim=-1)

        if all_obs.shape[1] - 24 == 0:
            return all_obs
        
        return torch.nn.functional.pad(all_obs, (0, 24-all_obs.shape[1]))
            

        if False and self.custom_box:
            # print('here')
            
            block_ids_int32 = torch.tensor([self.gym.find_actor_index(self.envs[i], world_cfg.movable_block.name, gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)
            
            mbox_actor_handles = [self.gym.find_actor_handle(self.envs[i], world_cfg.movable_block.name) for i in range(self.num_envs)]            
            block_ids_int32_contact = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs, mbox_actor_handles)], dtype=torch.long, device=self.device)
            

            
            # print(block_ids_int32, block_ids_int32_contact)  

            # print('here')

            obs = torch.cat([(self.all_root_states[block_ids_int32, :2] - self.env_origins[:, :2]).clone(), self.all_root_states[block_ids_int32, 3:7].clone(), self.block_size.clone()], dim=-1)
            
            block_contact_buf = (torch.linalg.norm(self.all_contact_forces[block_ids_int32_contact, :2], dim=-1) > 1.).view(-1, 1)
            self.block_contact_ctr = (~block_contact_buf) * self.block_contact_ctr
            self.block_contact_buf = self.block_contact_ctr < self.contact_memory_time
            self.block_contact_ctr[self.block_contact_ctr < self.contact_memory_time] += 1

            # print('movable', block_contact_buf[0], self.block_contact_ctr[0], self.block_contact_buf[0])

            see_obs = self.variables['all_obs_ids']

            extra_contact_obs = self.block_weight.clone() # * (self.block_contact_buf | see_obs.view(-1, 1)).view(-1, 1)
            obs = torch.cat([obs, extra_contact_obs], dim=-1)
            obs *= (self.block_contact_buf | see_obs.view(-1, 1)).view(-1, 1)

            if world_cfg.fixed_block.add_to_obs:
                fixed_block_ids = torch.tensor([self.gym.find_actor_index(self.envs[i], world_cfg.fixed_block.name, gymapi.DOMAIN_SIM) for i in range(self.num_envs)], dtype=torch.long, device=self.device)

                fbox_actor_handles = [self.gym.find_actor_handle(self.envs[i], world_cfg.movable_block.name) for i in range(self.num_envs)]            
                fixed_block_ids_contact = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs, fbox_actor_handles)], dtype=torch.long, device=self.device)



                fixed_block_contact_buf = (torch.linalg.norm(self.all_contact_forces[fixed_block_ids_contact, :2], dim=-1) > 1.).view(-1, 1)
                self.fixed_block_contact_ctr = (~fixed_block_contact_buf) * self.fixed_block_contact_ctr
                self.fixed_block_contact_buf = self.fixed_block_contact_ctr < self.contact_memory_time
                self.fixed_block_contact_ctr[self.fixed_block_contact_ctr < self.contact_memory_time] += 1
                

                fixed_block_obs = torch.cat([(self.all_root_states[fixed_block_ids, :2] - self.env_origins[:, :2]).clone(), self.fixed_block_size.clone()], dim=-1)
                self.fixed_block_contact_buf |= see_obs.view(-1, 1)
                fixed_block_obs *= self.fixed_block_contact_buf.int()

                obs = torch.cat([obs, fixed_block_obs], dim=-1)

            return obs
        # if world_cfg.fixed_block.add_to_obs:
        #     return torch.zeros(self.num_envs, 13, device=self.device)
        
        # return torch.zeros(self.num_envs, 9, device=self.device)