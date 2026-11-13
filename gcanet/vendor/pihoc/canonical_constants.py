import torch
import numpy as np

HUMAN_VIEW_DICT = { 
    '4MV-Z_Vitru': 
                    {'order': 'fix',
                    'num_vertices': 6890,
                    'grid_size': np.array([4, 1, 1]),
                    'mask_size': 1024,
                    'folder': 'hcontact_vitruvian',
                    'pixel_to_vertex': 'pixel_to_vertex_map_1024.npz',
                    'bary_coords': 'bary_coords_map_1024.npz',
                    'contact_annot_f': 'contact_label_objectwise.npy',
                    'body_parts_annot_f': 'body_parts_objectwise.pkl',
                    'names': np.array([[['topfront']],
                                       [['bottomfront']],
                                       [['topback']],
                                       [['bottomback']]]),
                    'ignore_keywords': [],
                    'cam_params': {
                        'topfront':    [2., 45., 315., 0., 0.],
                        'bottomfront': [2., 315., 315., 0., 0.3],
                        'topback':     [2., 45., 135., 0., 0.],
                        'bottomback':  [2., 315., 135, 0., 0.3]
                    }
                }, 
}


