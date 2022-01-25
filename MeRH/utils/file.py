from gzip import open as gopen
from io import SEEK_END
from json import loads
from os import fstat, getcwd, mkdir, remove
from os.path import basename, exists, getsize, splitext
from shutil import rmtree
from struct import unpack
from traceback import print_exc
from typing import Optional
from zipfile import ZipFile
from zlib import decompressobj

from requests import get
from tqdm import tqdm

from .switch import Switch

cwd = getcwd()

class Directory:
    def __init__(self, path: str):
        self.path = f"{cwd}/{path}"

    def __str__(self) -> str:
        try:
            mkdir(self.path)
        except:
            pass
        return self.path

LOGS = Directory("logs")
DATASET = Directory("dataset")
EMBEDDINGS = Directory("embeddings")

def exc_remove(*args):
    while True:
        try:
            for path in args:
                try:
                    remove(path)
                except:
                    pass
                rmtree(path, ignore_errors=True)
            print_exc()
            raise Exception
        except KeyboardInterrupt:
            pass

# https://stackoverflow.com/questions/22676/how-to-download-a-file-over-http
def download(url, parent_path: str, fname: Optional[str] = None, chunk_size: int = 1048576):
    try:
        fname = f"{basename(url) if fname is None else fname}"
        if exists(path := f"{parent_path}/{fname}"):
            print(f"{fname} exists, skipping...")
        else:
            resp = get(url, stream=True)
            resp.raise_for_status()
            print(f"Downloading {fname}...")
            with open(path, "wb") as f:
                with tqdm(total=int(resp.headers["Content-Length"])) as pbar:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        f.write(chunk)
                        pbar.update(len(chunk))
    except:
        exc_remove(path)

estimate_size_func = Switch()

# https://stackoverflow.com/questions/54329172/how-to-get-uncompressed-size-of-a-4gb-gz-file-in-python
@estimate_size_func(".gz")
def estimate_uncompressed_gz_size(filename: str) -> int:
    # From the input file, get some data:
    # - the 32 LSB from the gzip stream
    # - 1MB sample of compressed data
    # - compressed file size
    with open(filename, "rb") as gz_in:
        sample = gz_in.read(1000000)
        gz_in.seek(-4, SEEK_END)
        lsb = unpack('I', gz_in.read(4))[0]
        file_size = fstat(gz_in.fileno()).st_size

    # Estimate the total size by decompressing the sample to get the
    # compression ratio so we can extrapolate the uncompressed size
    # using the compression ratio and the real file size
    dobj = decompressobj(31)
    d_sample = dobj.decompress(sample)

    compressed_len = len(sample) - len(dobj.unconsumed_tail)
    decompressed_len = len(d_sample)

    estimate = file_size * decompressed_len // compressed_len

    # 32 LSB to zero
    mask = ~0xFFFFFFFF

    # Kill the 32 LSB to be substituted by the data read from the file
    adjusted_estimate = (estimate & mask) | lsb

    return adjusted_estimate

unzip_func = Switch()

@unzip_func(".gz")
def unzip_gzip(src_path: str, dst_path: str, chunk_size: int):
    with gopen(src_path, "rb") as src:
        with open(dst_path, "wb") as dst:
            with tqdm(total=estimate_uncompressed_gz_size(src_path)) as pbar:
                while chunk := src.read(chunk_size):
                    dst.write(chunk)
                    pbar.update(len(chunk))

# https://stackoverflow.com/questions/4006970/monitor-zip-file-extraction-python
@unzip_func(".zip")
def unzip_zip(src_path: str, dst_path: str, chunk_size: int):
    with ZipFile(src_path, "r") as src:
        dir_dst_path, basename_dst_path = dst_path.rsplit("/", 1)
        basename_dst_path += "/"
        for src_info in src.infolist():
            if src_info.filename == basename_dst_path:
                dst_path = dir_dst_path
                break
        else:
            mkdir(dst_path)
        with tqdm(total=sum(src_info.file_size for src_info in src.infolist())) as pbar:
            for src_info in src.infolist():
                if (fname := src_info.filename).endswith("/"):
                    mkdir(f"{dst_path}/{fname}")
                else:
                    with src.open(src_info) as src_file:
                        with open(f"{dst_path}/{fname}", "wb") as dst:
                            while chunk := src_file.read(chunk_size):
                                dst.write(chunk)
                                pbar.update(len(chunk))

def unzip(path: str, chunk_size: int = 1048576):
    try:
        dst_path, ext = splitext(path)
        if exists(dst_path):
            print(f"{basename(dst_path)} exists, skipping...")
        else:
            print(f"Unzipping {basename(path)}...")
            unzip_func[ext](path, dst_path, chunk_size)
    except:
        exc_remove(dst_path)

open_func = Switch({".gz": gopen})

def json_load(path: str, std: bool = True):
    print(f"Loading {basename(path)}...")
    ext = splitext(path)[1]
    with open_func.get(ext, open)(path, "rb") as f:
        # https://stackoverflow.com/questions/24890368/iterate-over-large-file-with-progress-indicator-in-python/24890594
        with tqdm(total=estimate_size_func.get(ext, getsize)(path)) as pbar:
            if std:
                while line := f.readline():
                    yield loads(line)
                    pbar.update(len(line))
            else:
                while line := f.readline():
                    yield eval(line)
                    pbar.update(len(line))
