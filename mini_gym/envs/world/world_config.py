import numpy as np

class world_cfg:
    CUSTOM_BLOCK = True
    class movable_block:
        name = 'movable_block'
        size_x_range = [0.1, 0.4]
        size_y_range = [1.0, 1.75] # [0.8, 1.5]
        pos_x_range = [.8, 1.5]
        pos_y_range = [-0.1, 0.1]
        block_density_range = [1, 6]

    class fixed_block:
        add_to_obs = True
        name = 'fixed_block'
        num_obs = 2
        size_x_range = [0.1, 0.4]
        size_y_range = [0.3, 0.8] # [0.5, 0.6] # [0.8, 1.5]
        pos_x_range = [1.8, 1.95]
        pos_y_range = [-0.5, 0.5]

INPLAY_ASSETS = [
        # {
        #     'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'],
        #     'size': [[lambda:0.5, lambda:0.8, lambda:0.5], [lambda:0.3, lambda:0.4, lambda: 0.5], [lambda:0.25, lambda:0.9, lambda:0.5]],
        #     'pos': [[1.5, 0.5, 0.25], [2.25, -0.15, 0.25], [1.5, -0.4, 0.25]],
        #     'density': [10000, 10000, 8]
        # },
        # {
        #     'name': ['fixed_asset_3', 'fixed_asset_4', 'movable_asset_2'],
        #     'size': [[0.5, 0.8, 0.5], [0.3, 0.4, 0.3], [0.25, 0.9, 0.5]],
        #     'pos': [[1.5, 0.5, 0.25], [2.25, -0.6, 0.15], [1.5, -0.4, 0.25]],
        #     'density': [10000, 10000, 3]
        # },
        
        # {
        #     'name': ['fixed_asset_5',  'movable_asset_3'],
        #     'size': [[0.5, 0.8, 0.5], [0.25, 0.9, 0.5]],
        #     'pos': [[1.5, 0.5, 0.25], [1.3, -0.4, 0.25]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'],
        #     'size': [[lambda:0.5, lambda:0.8, lambda:0.5], [lambda:0.3, lambda:0.4, lambda: 0.5], [lambda:0.25, lambda:0.9, lambda:0.5]],
        #     'pos': [[1.5, -0.5, 0.25], [2.25, 0.15, 0.15], [1.5, 0.4, 0.25]],
        #     'density': [10000, 10000, 8]
        # },
        # {
        #     'name': ['fixed_asset_8', 'fixed_asset_9', 'movable_asset_5'],
        #     'size': [[lambda:0.5, lambda:0.8, lambda:0.5], [lambda:0.3, lambda:0.4, lambda: 0.5], [lambda:0.25, lambda:0.9, lambda:0.5]],
        #     'pos': [[1.5, -0.5, 0.25], [2.25, 0.6, 0.25], [1.5, 0.4, 0.25]],
        #     'density': [10000, 10000, 3]
        # },
        
        # {
        #     'name': ['fixed_asset_10',  'movable_asset_6'],
        #     'size': [[lambda: 0.3, lambda:0.8, lambda: 0.3], [lambda:0.25, lambda:0.9, lambda:0.5]],
        #     'pos': [[1.5, -0.5, 0.25], [1.3, 0.4, 0.25]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_11', 'fixed_asset_12', 'movable_asset_7'],
        #     'size': [[0.5, 0.8, 0.5], [0.5, 0.6, 0.3], [0.25, 1.5, 0.3]],
        #     'pos': [[1.3, 0.5, 0.25], [2.1, -0.5, 0.15], [.6, 0.0, 0.15]],
        #     'density': [10000, 10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_13', 'fixed_asset_14', 'movable_asset_8'],
        #     'size': [[0.5, 0.8, 0.5], [0.5, 0.6, 0.3], [0.25, 1.7, 0.3]],
        #     'pos': [[1.3, -0.5, 0.25], [2.1, 0.5, 0.15], [.6, 0.0, 0.15]],
        #     'density': [10000, 10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_15', 'movable_asset_9'],
        #     'size': [[0.5, 0.8, 0.5], [0.25, 1.5, 0.3]],
        #     'pos': [[1.3, -0.5, 0.25], [.6, 0.0, 0.15]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_16', 'movable_asset_10'],
        #     'size': [[0.5, 0.6, 0.3], [0.25, 1.5, 0.3]],
        #     'pos': [[2.1, 0.5, 0.15], [.6, 0.0, 0.15]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_17', 'movable_asset_11'],
        #     'size': [[0.5, 0.8, 0.5], [0.25, 1.5, 0.3]],
        #     'pos': [[1.3, 0.5, 0.25], [.6, 0.0, 0.15]],
        #     'density': [10000, 3]
        # },

        # {
        #     'name': ['fixed_asset_18', 'movable_asset_12'],
        #     'size': [[0.5, 0.6, 0.3], [0.25, 1.5, 0.3]],
        #     'pos': [[2.1, -0.5, 0.15], [.6, 0.0, 0.15]], 
        #     'density': [10000, 3]
        # },

        {
            'name': ['fb_three_fixed_block_2', 'fb_three_fixed_block_1', 'fb_three_movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        # {
        #     'name': ['fb_three_fixed_block_4', 'fb_three_fixed_block_3', 'fb_three_movable_block_2'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },

        {
            'name': ['fb_three_fixed_block_6', 'fb_three_fixed_block_5', 'fb_three_movable_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['fb_three_fixed_block_8', 'fb_three_fixed_block_7', 'fb_three_movable_block_4'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        {
            'name': ['bb_three_fixed_block_2', 'bb_three_fixed_block_1', 'bb_three_movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.2]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.2]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },

        # {
        #     'name': ['bb_three_fixed_block_4', 'bb_three_fixed_block_3', 'bb_three_movable_block_2'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 10000, 8][::-1]
        # },
        
        {
            'name': ['bb_three_fixed_block_6', 'bb_three_fixed_block_5', 'bb_three_movable_block_3'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        {
            'name': ['bb_three_fixed_block_8', 'bb_three_fixed_block_7', 'bb_three_movable_block_4'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.0, 1.7]), 2), lambda: 0.4], [lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.0]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1],
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 10000, 8][::-1]
        },
        

        {
            'name': ['fixed_block_1', 'movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        },

        {
            'name': ['fixed_block_2', 'movable_block_2'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.7]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        },

        # {
        #     'name': ['fixed_block_2', 'movable_block_2'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), lambda: round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 8][::-1]
        # },

        # {
        #     'name': ['fixed_block_2', 'movable_block_2'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), lambda: round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), lambda: 0.4]][::-1],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 8][::-1]
        # },

        # {
        #     'name': ['movable_block_1'],
        #     'size': [[round(np.random.uniform(*[0.3, 0.3]), 2), round(np.random.uniform(*world_cfg.movable_block.size_y_range), 2), 0.4]],
        #     'pos': [[round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]],
        #     'density': [3]
        # },
        # {
        #     'name': ['fixed_block_test'],
        #     'size': [[lambda : round(np.random.uniform(*world_cfg.fixed_block.size_x_range), 2), lambda: round(np.random.uniform(*world_cfg.fixed_block.size_y_range), 2), lambda: 0.4]],
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2]],
        #     'density': [10000]
        # },

        # {
        #     'name': ['fixed_block_test'],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.7]), 2), lambda: 0.4]],
        #     'pos': [[round(np.random.uniform(*[world_cfg.movable_block.pos_x_range[0], world_cfg.movable_block.pos_x_range[1]]), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2]],
        #     'density': [10000]
        # },

        {
            'name': ['fixed_block_test'],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.7]), 2), lambda: 0.4]],
            'pos': [[round(np.random.uniform(*[world_cfg.movable_block.pos_x_range[0], world_cfg.movable_block.pos_x_range[1]]), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2]],
            'density': [10000]
        },

        {
            'name': ['movable_block_test'],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.7]), 2), lambda: 0.4]],
            'pos': [[round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]],
            'density': [8]
        },
    ]

EVAL_INPLAY_ASSETS = [

        {
            'name': ['fixed_asset_1', 'fixed_asset_2', 'movable_asset_1'],
            'size': [[lambda:0.3, lambda:0.8, lambda: 0.4], [lambda:0.3, lambda:0.4, lambda: 0.4], [lambda:0.3, lambda:0.9, lambda: 0.4]],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.15, 0.2], [1.5, -0.4, 0.2]],
            'density': [10000, 10000, 8]
        },
        
        {
            'name': ['fixed_asset_3', 'fixed_asset_4', 'movable_asset_2'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.4, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, 0.5, 0.2], [2.25, -0.6, 0.2], [1.5, -0.4, 0.2]],
            'density': [10000, 10000, 8]
        },

        {
            'name': ['fixed_asset_6', 'fixed_asset_7', 'movable_asset_4'],
            'size': [[lambda:0.3, lambda:0.8, lambda: 0.4], [lambda:0.3, lambda:0.4, lambda: 0.4], [lambda:0.3, lambda:0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.15, 0.2], [1.5, 0.4, 0.2]],
            'density': [10000, 10000, 8]
        },

        {
            'name': ['fixed_asset_8', 'fixed_asset_9', 'movable_asset_5'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.4, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [2.25, 0.6, 0.2], [1.5, 0.4, 0.2]],
            'density': [10000, 10000, 8]
        },
        
        {
            'name': ['fixed_asset_10',  'movable_asset_6'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.9, lambda: 0.4]],
            'pos': [[1.5, -0.5, 0.2], [1.3, 0.4, 0.2]],
            'density': [10000, 8]
        },

        {
            'name': ['fixed_asset_11', 'fixed_asset_12', 'movable_asset_7'],
            'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.6, lambda: 0.4]],
            'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
            'density': [10000, 10000, 8]
        },

        {
            'name': ['fixed_asset_13', 'fixed_asset_14', 'movable_asset_8'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.6, lambda: 0.4]],
            'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
            'density': [10000, 10000, 8]
        },

        # {
        #     'name': ['fixed_asset_11_dup_1', 'fixed_asset_12_dup_1', 'movable_asset_7_dup_1'],
        #     'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_13_dup_1', 'fixed_asset_14_dup_1', 'movable_asset_8_dup_1'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.7, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_11_dup_2', 'fixed_asset_12_dup_2', 'movable_asset_7_dup_2'],
        #     'size':  [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
        #     'pos': [[1.3, 0.5, 0.2], [2.1, -0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        # {
        #     'name': ['fixed_asset_13_dup_2', 'fixed_asset_14_dup_2', 'movable_asset_8_dup_2'],
        #     'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.7, lambda: 0.4]],
        #     'pos': [[1.3, -0.5, 0.2], [2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
        #     'density': [10000, 10000, 8]
        # },

        {
            'name': ['fixed_asset_15', 'movable_asset_9'],
            'size': [[lambda: 0.3, lambda: 0.8, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
            'pos': [[1.3, -0.5, 0.2], [.6, 0.0, 0.2]],
            'density': [10000, 8]
        },

        {
            'name': ['fixed_asset_16', 'movable_asset_10'],
            'size': [[lambda: 0.3, lambda: 0.6, lambda: 0.4], [lambda: 0.3, lambda: 1.5, lambda: 0.4]],
            'pos': [[2.1, 0.5, 0.2], [.6, 0.0, 0.2]],
            'density': [10000, 8]
        },

        {
            'name': ['fixed_block_1', 'movable_block_1'][::-1],
            'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 0.8]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[1.3, 1.7]), 2), lambda: 0.4]][::-1], 
            'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), 0.2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
            'density': [10000, 8][::-1]
        }

        # {
        #     'name': ['fixed_block_1', 'movable_block_1'][::-1],
        #     'size': [[lambda : round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[0.3, 1.5]), 2), lambda: 0.4], [lambda: round(np.random.uniform(*[0.3, 0.3]), 2), lambda: round(np.random.uniform(*[.3, 1.7]), 2), lambda: 0.4]][::-1], 
        #     'pos': [[round(np.random.uniform(*world_cfg.fixed_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.fixed_block.pos_y_range), 2), .2], [round(np.random.uniform(*world_cfg.movable_block.pos_x_range), 2), round(np.random.uniform(*world_cfg.movable_block.pos_y_range), 2), 0.2]][::-1],
        #     'density': [10000, 8][::-1]
        # }
    ]

# EVAL_INPLAY_ASSETS = [*INPLAY_ASSETS]
INPLAY_ASSETS = [*EVAL_INPLAY_ASSETS]