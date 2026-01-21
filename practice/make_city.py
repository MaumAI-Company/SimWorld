import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent))
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import UnrealCV
import os

ucv = UnrealCV()
communicator = Communicator(ucv)
communicator.generate_world(os.path.abspath('./examples/output/progen_world.json'), os.path.abspath('./data/ue_assets.json'))

