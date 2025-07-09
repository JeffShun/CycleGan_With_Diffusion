import os
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
from tensorboardX import SummaryWriter

def get_writer(tensorboad_dir):
    os.makedirs(tensorboad_dir, exist_ok=True)
    writer = SummaryWriter(tensorboad_dir)
    return writer
