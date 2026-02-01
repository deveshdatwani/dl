import os, tempfile, torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def atomic_save(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fd, tmp = tempfile.mkstemp()
    os.close(fd)
    torch.save(obj, tmp)
    os.replace(tmp, path)
