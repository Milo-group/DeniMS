
import io, lmdb, pickle, torch
from torch_geometric.data import Data

LMDB_META_KEY = b"__meta__"

def _dumps(obj) -> bytes:
    buf = io.BytesIO()
    torch.save(obj, buf)           # uses pickle by default; matches torch_geometric Data semantics
    return buf.getvalue()

def _loads(b: bytes):
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)

def open_env(path: str, map_size: int = 32 * 1024**3, readonly: bool = False) -> lmdb.Environment:
    # map_size default ~16GB (adjust if needed)
    return lmdb.open(path, map_size=map_size, subdir=True, readonly=readonly, lock=not readonly, readahead=readonly, max_readers=2048)

def write_meta(txn, meta: dict):
    txn.put(LMDB_META_KEY, pickle.dumps(meta))

def read_meta(txn) -> dict:
    raw = txn.get(LMDB_META_KEY)
    if raw is None:
        raise RuntimeError("LMDB meta not found")
    return pickle.loads(raw)
