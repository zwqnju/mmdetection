import sys
from mmcv import Config

config_file = sys.argv[1]
cfg = Config.fromfile(config_file)
print(f'Config:\n{cfg.pretty_text}')
