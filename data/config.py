# config.py

cfg = {
    'name': 'Retinaface',
    'min_sizes': [[32, 64], [128, 256], [512, 1024]],
    'steps': [16, 32, 64],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True
}
