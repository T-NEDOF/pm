import os

# from neural_network_model.config import config
from packages.neural_network_model.neural_network_model.config import config


with open(os.path.join(config.PACKAGE_ROOT, 'VERSION')) as version_file:
    __version__ = version_file.read().strip()
