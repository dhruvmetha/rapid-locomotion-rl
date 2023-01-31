from isaacgym import gymtorch, gymapi, gymutil
from isaacgym.torch_utils import *

assert gymtorch
import torch
import time
import random

from high_level_policy import *
from high_level_policy import task_inplay
from mini_gym.envs.world.world_config import *

np.random.seed(42)
torch.manual_seed(42)

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
    def __init__(self, cfg, sim, gym, device, train_ratio=0.95) -> None:
        """
        Creates the world environment objects like obstacles, walls, etc.
        """
        self.envs = None # gets initialized in post_create_world
        self.gym = gym
        self.sim = sim
        self.cfg = cfg
        self.num_envs = self.cfg.env.num_envs # gets initialized in post_create_world
        self.num_train_envs = max(1, int(self.num_envs*train_ratio)) # gets initialized in post_create_world
        self.num_eval_envs = self.num_envs - self.num_train_envs # gets initialized in post_create_world
        self.device = device


        self.custom_box = world_cfg.CUSTOM_BLOCK


        # initialize buffers and variables needed for the world actors
        self.handles = {}
        self.env_assets_map = {}
        self.env_actor_indices_map = {}
        self.all_actor_base_postions = {}
        self.variables = None

        self.contact_memory_time = 20
        self.reset_timer_count = 30

        tasks = {
            'task_0': TASK_0,
            'task_1': TASK_1,
            'task_2': TASK_2,
        }

        self.inplay_assets = INPLAY_ASSETS
        self.eval_inplay_assets =  EVAL_INPLAY_ASSETS # INPLAY_ASSETS

        self.inplay = {}

        self.world_types_success = {}


        self.train_eval_assets = {}

        self.base_assets = []
        self.asset_counts = 0
        self.asset_idx_map = {}
        self.idx_asset_map = {}
        self.env_assetname_map = {i: {} for i in range(self.num_envs)}
        self.env_assetname_bool_map = {i: {} for i in range(self.num_envs)}

        self.world_types = 0
        self.eval_world_types = 0
        
        idx_ctr = 0
        for i in self.inplay_assets:
            self.world_types_success[self.world_types] = 0
            self.world_types += 1
            self.asset_counts += len(i['name'])
            for j in i['name']:
                self.train_eval_assets[j] = False
                self.asset_idx_map[j] = idx_ctr
                self.idx_asset_map[idx_ctr] = j
                idx_ctr += 1


        for i in self.eval_inplay_assets:
            self.asset_counts += len(i['name'])
            self.eval_world_types += 1
            for j in i['name']:
                name_eval = j + '_eval'
                self.train_eval_assets[name_eval] = True
                self.asset_idx_map[name_eval] = idx_ctr
                self.idx_asset_map[idx_ctr] = name_eval
                idx_ctr += 1

        self.block_size = torch.zeros((self.num_envs, self.asset_counts, 2), device=self.device, dtype=torch.float, requires_grad=False)
        self.block_weight = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.float, requires_grad=False)

        self.block_contact_buf = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.block_contact_ctr = torch.zeros((self.num_envs, self.asset_counts, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.fixed_block_contact_buf = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.bool, requires_grad=False)
        self.fixed_block_contact_ctr = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.int, requires_grad=False) + self.contact_memory_time + 1

        self.inplay_env_world = torch.zeros((self.num_envs, self.world_types), device=self.device, dtype=torch.bool, requires_grad=False)
        self.env_world_success = torch.zeros(self.world_types, device=self.device, dtype=torch.int, requires_grad=False)
        self.env_world_counts = torch.zeros(self.world_types, device=self.device, dtype=torch.int, requires_grad=False)

        self.eval_worlds = torch.zeros(self.num_eval_envs, device=self.device, dtype=torch.int, requires_grad=False)
        self.total_eval_worlds = (self.num_eval_envs//self.eval_world_types) * self.eval_world_types
        
        if self.total_eval_worlds == 0:
             self.eval_worlds[:] = torch.randint(0, self.eval_world_types, (self.num_eval_envs,))
        else:
            self.eval_worlds[:self.total_eval_worlds] = torch.arange(0, self.eval_world_types).view(-1, 1).repeat(1, self.num_eval_envs//self.eval_world_types).view(-1)
            self.eval_worlds[self.total_eval_worlds:] = torch.randint(0, self.eval_world_types, (self.num_eval_envs - self.total_eval_worlds,))
            
        self.world_sampling_dist = torch.zeros(self.world_types, device=self.device, dtype=torch.float, requires_grad=False) + 1/self.world_types

        # self.obs = torch.zeros((self.num_envs, 24), device=self.device, dtype=torch.bool, requires_grad=False)

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

        self.per_rect = PER_RECT

        self.base_assets = [AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)]

        self.env_asset_ctr = torch.zeros(self.num_envs, self.asset_counts, self.per_rect, dtype=torch.long, device=self.device)
        self.env_asset_ctr[:, :, :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)
        self.env_asset_bool = torch.zeros(self.num_envs, self.asset_counts, 1, dtype=torch.bool, device=self.device)
        self.all_train_ids = torch.arange(0, self.num_train_envs, 1, dtype=torch.long, device=self.device).view(-1, 1)
        self.all_eval_ids = torch.arange(self.num_train_envs, self.num_envs, 1, dtype=torch.long, device=self.device).view(-1, 1)


        self.reset_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) + self.reset_timer_count
        self.last_obs = None

    def define_world(self, env_id):
        """
        define the world configuration and it's assets
        """
        
        assets_container = []

        assets_inplay = self.inplay_assets if env_id < self.num_train_envs else self.eval_inplay_assets

        for i in assets_inplay:
            gym_assets = []
            asset_names = []
            asset_pos = []

            for idx, name in enumerate(i['name']):
                asset_options = gymapi.AssetOptions()
                asset_options.disable_gravity = False
                asset_options.fix_base_link = False

                sizes = [j() for j in i['size'][idx]]
                volume = np.prod(sizes)
                density = 1./volume

                if 'fixed' in name:
                    asset_options.density = i['density'][idx]
                else:
                    asset_options.density = density

                gym_assets.append(self.gym.create_box(self.sim, *sizes, asset_options))
                final_name = name if env_id < self.num_train_envs else name + '_eval'
                
                self.block_size[env_id, self.asset_idx_map[final_name], :] = torch.tensor(sizes[:2])
                asset_names.append(final_name)
                
                asset_pos.append(i['pos'][idx])
            assets_container.append([AssetDef(asset, name, pos) for asset, name, pos in zip(gym_assets, asset_names, asset_pos)])

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
            # print(name)
            if not self.train_eval_assets[name]:
                self.asset_root_ids[name] = torch.tensor([self.gym.find_actor_index(self.envs[i], name, gymapi.DOMAIN_SIM) for i in range(self.num_train_envs)], dtype=torch.long, device=self.device)
                block_actor_handles = [self.gym.find_actor_handle(self.envs[i], name) for i in range(self.num_train_envs)]            
                self.asset_contact_ids[name] = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs[:self.num_train_envs], block_actor_handles)], dtype=torch.long, device=self.device)
            else:
                self.asset_root_ids[name] = torch.tensor([self.gym.find_actor_index(self.envs[i], name, gymapi.DOMAIN_SIM) for i in range(self.num_train_envs, self.num_envs)], dtype=torch.long, device=self.device)
                block_actor_handles = [self.gym.find_actor_handle(self.envs[i], name) for i in range(self.num_train_envs, self.num_envs)]            
                self.asset_contact_ids[name] = torch.tensor([self.gym.find_actor_rigid_body_index(i, j, "box", gymapi.DOMAIN_SIM) for i, j in zip(self.envs[self.num_train_envs:self.num_envs], block_actor_handles)], dtype=torch.long, device=self.device)

     
    def init_buffers(self, **kwargs):
        """
        world buffers for actor root states, rb, contacts
        """
        self.all_root_states = kwargs['root_states']
        self.all_dof_state = kwargs['dof_states']
        self.all_rigid_body_state = kwargs['rigid_body_states']
        self.all_contact_forces = kwargs['contact_forces']
        return

    def _get_random_idx(self, env_id):
        if ADAPTIVE_SAMPLE_ENVS and (env_id < self.num_train_envs):
            return torch.multinomial(self.world_sampling_dist, 1)
        else:
            assets_container = self.env_assets_map[env_id]
            if env_id >= self.num_train_envs:
                if self.total_eval_worlds == 0:
                    return torch.randint(0, len(assets_container), (1,))
                else:
                    if env_id > (self.num_train_envs + self.total_eval_worlds-1):
                        return torch.randint(0, len(assets_container), (1,))
                    else:
                        return self.eval_worlds[env_id - self.num_train_envs]
            return torch.randint(0, len(assets_container), (1,))
        
    def reset_world(self, env_ids, _):
        """
        reset the world actors in the environment
        """
        # TODO: precompute these lists for more efficiency in post_create_world

        actor_indices = []
        base_positions = []
        env_origins = []
        env_asset_bool = []
        self.env_asset_bool[env_ids, :, :] = False

        for env_id in env_ids:
            env_id = env_id.item()
            env_handle = self.envs[env_id]
            assets_container = self.env_assets_map[env_id]

            random_idx = self._get_random_idx(env_id)
            self.inplay_env_world[env_id, :]  = False
            if env_id < self.num_train_envs:
                self.inplay_env_world[env_id, random_idx] = True
                self.inplay[env_id] = random_idx
            in_indices = []
            assets_marked = []

            current_ctr = 0

            for idx, asset_container in enumerate(assets_container):
                mv_size = None
                movable_bp = None
                movable_asset_name = None 
                fixed_bp = None
                fb_three_bool = [False, False, False]
                bb_three_bool = [False, False, False]

                for asset in asset_container:
                    actor_indices.append(self.gym.find_actor_index(env_handle, asset.name, gymapi.DOMAIN_SIM))
                    env_origins.append(self.env_origins[env_id])
                    assets_marked.append(asset.name)
                    self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)

                    if idx == random_idx:

                        if asset.name.startswith('movable_block'):
                            self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(0, self.per_rect, dtype=torch.long, device=self.device)
                        elif asset.name.startswith('fixed_block'):
                            self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] = torch.arange(self.per_rect, int(self.per_rect*2), dtype=torch.long, device=self.device)


                        # self.env_asset_ctr[env_id, self.asset_idx_map[asset.name], :] += self.per_rect*current_ctr
                        
                        if asset.name.startswith('fb_three_mov'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                            mv_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            mv_y_range = (0.975 - mv_size_y/2)

                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]

                            movable_asset_name = asset.name
                            mv_size = (mv_size_x, mv_size_y)

                            base_positions.append(movable_bp)
                            fb_three_bool[0] = True

                        elif asset.name.startswith('fb_three_fix'):
                            
                            mv_x, mv_y, _ = movable_bp
                            mv_size_x, mv_size_y = mv_size
                            fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            if not fb_three_bool[1]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x-mv_size_x/2-fx_size_x/2-np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                fb_three_bool[1] = True


                            elif fb_three_bool[1] and not fb_three_bool[2]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                fb_three_bool[2] = True

                            base_positions.append(fixed_bp)

                        elif asset.name.startswith('bb_three_mov'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                            mv_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            mv_y_range = (0.975 - mv_size_y/2)

                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]

                            movable_asset_name = asset.name
                            mv_size = (mv_size_x, mv_size_y)

                            base_positions.append(movable_bp)
                            bb_three_bool[0] = True
                        
                        elif asset.name.startswith('bb_three_fix'):
                            
                            mv_x, mv_y, _ = movable_bp
                            mv_size_x, mv_size_y = mv_size
                            fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()

                            if not bb_three_bool[1]:
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                bb_three_bool[1] = True

                            elif bb_three_bool[1] and not bb_three_bool[2]:
                                bb_offset = fixed_bp[0]
                                fx_y_range = (0.975 - fx_size_y/2)
                                fixed_bp = [mv_x+bb_offset/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                bb_three_bool[2] = True


                            base_positions.append(fixed_bp)

                        elif asset.name == world_cfg.movable_block.name or asset.name.startswith('movable_block'):
                            if fixed_bp is not None:
                                raise "should do movable first"
                                exit()

                            mv_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()
                            mv_y_range = (0.975 - mv_size_y/2)
                            
                            movable_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-mv_y_range, mv_y_range]), 2), 0.2]
                            movable_asset_name = asset.name

                            base_positions.append(movable_bp)

                        elif asset.name == world_cfg.fixed_block.name or asset.name.startswith('fixed_block'):
                            fx_size_y = self.block_size[env_id, self.asset_idx_map[asset.name], 1].item()
                            fx_y_range = (0.975 - fx_size_y/2)
                            if movable_bp is not None:
                                mv_x, mv_y, _ = movable_bp
                                mv_size_x = self.block_size[env_id, self.asset_idx_map[movable_asset_name], 0]
                                fx_size_x = self.block_size[env_id, self.asset_idx_map[asset.name], 0].item()

                                # if np.random.uniform() < 0.5 and fx_size_y < 0.8:
                                #     fixed_bp = [mv_x-mv_size_x/2-fx_size_x/2-np.random.uniform(0, 0.2), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                                # else:
                                fixed_bp = [mv_x+mv_size_x/2+fx_size_x/2+np.random.uniform(0, 0.35), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                            else:
                                fixed_bp = [round(np.random.uniform(*[0.7, 2.0]), 2), round(np.random.uniform(*[-fx_y_range, fx_y_range]), 2), 0.2]
                            base_positions.append(fixed_bp)
                        
                        else:
                            base_positions.append(asset.base_position)
                    
                        # current_ctr += 1

                    else:
                        base_positions.append([bp - 100000.0 for bp in asset.base_position])
                    
                    
                    
            
        actor_indices = torch.tensor(actor_indices, dtype=torch.long, device='cuda:0')
        base_positions = torch.tensor(base_positions, dtype=torch.float32, device='cuda:0')

        
        env_origins = torch.vstack(env_origins)
        self.all_root_states[actor_indices, :3] = base_positions + env_origins
        self.all_root_states[actor_indices, 3:] = 0.
        self.all_root_states[actor_indices, 6] = 1. 
        self.block_contact_buf[env_ids, :, :] = False
        self.block_contact_ctr[env_ids, :, :] = self.contact_memory_time + 1
        self.reset_timer[env_ids] = self.reset_timer_count

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.all_root_states), gymtorch.unwrap_tensor(actor_indices.to(dtype=torch.int32)), len(actor_indices))

    def get_block_obs(self):

        ids = []
        block_actor_handles = []
        ids_contact = []
        all_obs = torch.zeros((self.num_envs, PER_RECT*2), dtype=torch.float, device=self.device)
        full_seen_obs = torch.zeros_like(all_obs)
        if self.last_obs is None:
            self.last_obs = torch.zeros_like(all_obs) 
        # self.env_asset_ctr[:, :] = torch.arange(0, 10, dtype=torch.long, device=self.device)
        asset_keys = list(self.asset_idx_map.keys())
        # random.shuffle(asset_keys)
        for _, name in enumerate(asset_keys):
            
            curr_env_ids = self.all_train_ids.view(-1)
            if self.train_eval_assets[name]:
                curr_env_ids = self.all_eval_ids.view(-1)
            if len(curr_env_ids) == 0:
                continue 
            
            ids = self.asset_root_ids[name]
            ids_contact = self.asset_contact_ids[name]

            movable_indicator = 0
            if 'mov' in name:
                movable_indicator = 1

            self.env_asset_bool[curr_env_ids, self.asset_idx_map[name], :] = ~(self.all_root_states[ids, 0] < -100).view(-1, 1)

            rot = self.all_root_states[ids, 3:7]
            angle = torch.atan2(2.0*(rot[:, 0]*rot[:, 1] + rot[:, 3]*rot[:, 2]), 1. - 2.*(rot[:, 1]*rot[:, 1] + rot[:, 2]*rot[:, 2]))

            # print(angle.shape, (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).shape)

            all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] += (torch.cat([torch.tensor([1, movable_indicator], device=self.device).repeat(len(curr_env_ids), 1), (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).clone(), angle.view(-1, 1), self.block_size[curr_env_ids, self.asset_idx_map[name], :].clone()], dim=-1) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]])

            full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] += (torch.cat([torch.tensor([1, movable_indicator], device=self.device).repeat(len(curr_env_ids), 1), (self.all_root_states[ids, :2] - self.env_origins[curr_env_ids, :2]).clone(), angle.view(-1, 1), self.block_size[curr_env_ids, self.asset_idx_map[name], :].clone()], dim=-1) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]])
            
            # if self.train_eval_assets[name] and self.env_asset_bool[curr_env_ids[0], self.asset_idx_map[name]]:
            #     if torch.linalg.norm(all_obs[curr_env_ids.view(-1, 1)[0], -8:]) > 0:
            #         print(self.all_train_ids, self.all_eval_ids)
            #         quit()

            contact_forces = self.all_contact_forces[ids_contact]
            block_contact_buf = (torch.linalg.norm(self.all_contact_forces[ids_contact, :2], dim=-1) > 0.1).view(-1, 1) * (self.reset_timer[curr_env_ids] == 0).view(-1, 1) # * (torch.linalg.norm(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] - self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]], dim=-1) > 0.05).view(-1, 1)

            one_touch_bcb = block_contact_buf | (self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :] & ONE_TOUCH_MAP)
            self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] = (~one_touch_bcb) * self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :]
            self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :] = self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] < self.contact_memory_time
            self.block_contact_ctr[curr_env_ids, self.asset_idx_map[name], :] += (1*self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :])
            
            all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]] *= (self.variables['full_info'] | self.block_contact_buf[curr_env_ids, self.asset_idx_map[name], :])

            # all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][:, 2:4]] * 

            # if not ONE_TOUCH_MAP:
            all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][:, :1]] *= block_contact_buf
            # print(all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][:, :1]], all_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][:, 4:5]])

            # self.env_asset_ctr[curr_env_ids, :] += 10*(self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]].view(-1, 1))

            ind = self.block_contact_buf[curr_env_ids, self.asset_idx_map[name]] * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]]
            # print(all_obs[curr_env_ids].shape)
            # print(all_obs[curr_env_ids[ind.view(-1)]])
            if torch.sum(ind) > 0:
                check_obs = all_obs[(curr_env_ids[ind.view(-1)]).view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][ind.view(-1), :]]
                checker = torch.sum(check_obs, dim=-1) == 0
                if checker.sum() > 0:
                    print("##################")
                    print(check_obs[checker])
            # print(all_obs[curr_env_ids[ind.view(-1)], self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][ind.view(-1), :]])
            # print(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:4].shape)
            # print(self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:4].shape)
            
            diff = torch.round(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:4] - self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]][:, 2:4], decimals=3)

            _change = (torch.linalg.norm(diff, dim=-1) > 0).view(-1, 1) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]].view(-1, 1) * (torch.sum(full_seen_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]], dim=-1).view(-1, 1) > 0) * (torch.sum(self.last_obs[curr_env_ids.view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]]], dim=-1).view(-1, 1) > 0)

            # print(_change.shape, ind.shape)
            
            # print(_change)
            # * (torch.sum(full_seen_obs[curr_env_ids, :], dim=-1) > 0) * self.env_asset_bool[curr_env_ids, self.asset_idx_map[name]].view(-1))

            if _change.sum() > 0:
                change_ind  = _change * ind
                # print(change_ind.shape)
                if change_ind.sum() > 0:
                    check_obs = all_obs[(curr_env_ids[change_ind.view(-1)]).view(-1, 1), self.env_asset_ctr[curr_env_ids, self.asset_idx_map[name]][change_ind.view(-1), :]]
                    checker = torch.sum(check_obs[:, 2:4], dim=-1) == 0
                    if checker.sum() > 0:
                        print("##################")
                        print(check_obs[checker])


            # # print((torch.linalg.norm(torch.round(full_seen_obs[curr_env_ids, 2:4] - self.last_obs[curr_env_ids, 2:4], decimals=2), dim=-1) > 0), torch.sum(self.last_obs[curr_env_ids, :], dim=-1), (torch.sum(self.last_obs[curr_env_ids, :], dim=-1) > 0))
            # # print(torch.round(full_seen_obs[curr_env_ids, 2:4] - self.last_obs[curr_env_ids, 2:4], decimals=3), full_seen_obs[curr_env_ids, 2:4], self.last_obs[curr_env_ids, 2:4], _change)
            
            # _no_obs = (torch.sum(all_obs[curr_env_ids][_change, 2:4], dim=-1) == 0)
            # _no_obs = (torch.sum(all_obs[curr_env_ids], dim=-1) == 0) * _change
            # print("############################")
            # print(_no_obs.shape)
            # if _no_obs.size(0) > 0:
            #     _no_obs *= (self.reset_timer[curr_env_ids][_change] == 0)
            #     print(_no_obs.shape)
            #     if _no_obs.sum() > 0:
            #         if 'movable' in name:
            #             print("############################")
            #             print(_no_obs)
            #             print((self.all_root_states[ids, 0]))
            #             print(self.env_asset_bool[curr_env_ids, self.asset_idx_map[name], :])
            #             print(name)
            #             # print(contact_forces.shape, ids_contact.shape, _no_obs.shape)
            #             print(contact_forces[_no_obs.nonzero()[:, 0]])
            #             print(block_contact_buf[_no_obs.nonzero()[:, 0]], self.block_contact_buf[_no_obs.nonzero()[:, 0], self.asset_idx_map[name], :])

            #             print(all_obs[curr_env_ids][_no_obs.nonzero()[:, 0], 2:5])

        if not self.variables['full_info']:
            all_obs *= (self.reset_timer == 0).view(-1, 1)
            self.reset_timer[:] -= (self.reset_timer[:] > 0).int()

        self.last_obs[:, :] = full_seen_obs[:, :]
        return all_obs, full_seen_obs