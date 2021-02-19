#/usr/bin/python3

"""This is an example dataset using Apache Arrow to match tensor record
 performance in pytorch
"""

import logging

from pathlib import Path
from PIL import Image
import six
from tqdm import tqdm
import lmdb
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImageNet(Dataset):
    """ LMDB based dataset which allows for quicker iteration
    compared to reading from disc.

    :param path_to_db: Path to lmdb dataset
    :param transform: Transform from PIL image to tensor
    """

    def __init__(self, path_to_db: str, transform=None):
        self.path_to_db = Path(path_to_db).expanduser()
        self.env = None
        self.length = 0
        self.keys = None
        self._init_db()
        self.transform = transform

    def _init_db(self):
        self.env = lmdb.open(
            str(self.path_to_db),
            subdir=self.path_to_db.is_dir(),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False
        )

        with self.env.begin(write=False) as txn:
            # self.length = txn.stat()['entries'] - 1
            self.length = pa.deserialize(txn.get(b'__len__'))
            self.keys = pa.deserialize(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def create_bookkeeping(self):
        """Create meta-dataset bookkeeping for the current

        This requires iterating over the entire dataset and should
        only be done once.
        """
        indices_to_labels = {i: self[i][1] for i in range(len(self))}
        labels_to_indices = defaultdict(list)
        for i, label in indices_to_labels.items():
            labels_to_indices[label].append(i)

        self.indices_to_labels = indices_to_labels
        self.labels_to_indices = labels_to_indices

    @classmethod
    def create_db(cls, root, outpath, write_frequency=5000):
        """Convert image folder to lmdb

        Convert a folder of the structure:
            root/class-name1/img1.png
            root/class-name1/img2.png
            ...
            root/class-name2/img1.png
            ...

        and converts it to a *lmdb* database with serialized
        images and indexed label. Labels are created by sorting
        classes and enumerating.

        :param root: Path to image-folder root
        :param outpath: Path to store lmdb file
        :param write_frequency: The rate at which processed images should be
            saved
        """

        def _raw_reader(path):
            with open(path, 'rb') as f:
                bin_data = f.read()
            return bin_data

        def _dumps_pyarrow(obj):
            return pa.serialize(obj).to_buffer()

        directory = Path(root).expanduser()
        logger.info("Loading dataset from %s", directory)
        dataset = ImageFolder(directory, loader=_raw_reader)
        logger.info("Loaded dataset with size %d", len(dataset))
        data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

        lmdb_path = Path(outpath).expanduser()
        isdir = lmdb_path.is_dir()

        logger.info("Generate LMDB to %s", str(lmdb_path))
        db = lmdb.open(str(lmdb_path), subdir=isdir,
                    map_size=1099511627776 * 4, readonly=False,
                    meminit=False, map_async=True)

        txn = db.begin(write=True)
        for idx, data in tqdm(enumerate(data_loader), desc='Loading data'):
            image, label = data[0]
            txn.put(u'{}'.format(idx).encode('ascii'), _dumps_pyarrow((image, label)))
            if idx % write_frequency == 0:
                logger.info("Wrote [%d/%d]", idx, len(data_loader))
                txn.commit()
                txn = db.begin(write=True)

        # Finish iterating through dataset
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(len(data_loader))]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', _dumps_pyarrow(keys))
            txn.put(b'__len__', _dumps_pyarrow(len(keys)))

        logger.info("Flushing database")
        db.sync()
        db.close()

    def __getstate__(self):
        # The lmdb can not be pickled which creates problem
        # when using DistributedDataParallel. Instead we discard it
        # and reload it when we initialze the clone of the dataset
        attributes = self.__dict__.copy()
        del attributes['env']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        # NOTE: We need to reconnect to the database after unpickling.
        # This can create problems if using DistributedDataParallel
        # over mulitple machines. Should work fine normally though.
        self._init_db()

    def __getitem__(self, index):
        img, target = None, None
        env = self.env

        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
            unpacked = pa.deserialize(byteflow)

        # load image
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        if self.transform is not None:
            img = self.transform(img)

        return img, target

if __name__ == '__main__':
    breakpoint()
    # ImageNet.create_db("/mnt/hdd1/data/image_net/validation/structured/", "/mnt/hdd1/data/image_net/validation.lmdb", 100000)
    ds = ImageNet("/mnt/hdd1/data/image_net/validation.lmdb")
    test = ds[0]
    breakpoint()
