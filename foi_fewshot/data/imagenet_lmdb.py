#/usr/bin/python3

"""This is an example dataset using Apache Arrow to create a fast

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

logger = logging.getLogger(__name__)

class ImageNet(Dataset):
    """ImageNet
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
            self.path_to_db,
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


    @classmethod
    def create_bookkeeping(cls, dataset):
        indices_to_labels = {i: dataset[i][1] for i in range(len(dataset))}
        labels_to_indices = defaultdict(list)
        for i, label in indices_to_labels.items():
            labels_to_indices[label].append(i)
        return indices_to_labels, labels_to_indices

    @classmethod
    def create_db(cls, path, outpath, write_frequency=5000):

        def _raw_reader():
            with open(path, 'rb') as f:
                bin_data = f.read()
            return bin_data

        def _dumps_pyarrow(obj):
            return pa.serialize(obj).to_buffer()

        directory = Path(path).expanduser()
        logger.info("Loading dataset from %s" % directory)
        dataset = ImageFolder(directory, loader=_raw_reader)
        data_loader = DataLoader(dataset, num_workers=16, collate_fn=lambda x: x)

        lmdb_path = Path(outpath).expanduser()
        isdir = lmdb_path.is_dir()

        logger.info("Generate LMDB to %s" % lmdb_path)
        db = lmdb.open(lmdb_path, subdir=isdir,
                    map_size=1099511627776 * 2, readonly=False,
                    meminit=False, map_async=True)

        txn = db.begin(write=True)
        for idx, data in tqdm(enumerate(data_loader), desc='Loadng data'):
            image, label = data[0]
            txn.put(u'{}'.format(idx).encode('ascii'), _dumps_pyarrow((image, label)))
            if idx % write_frequency == 0:
                print("[%d/%d]" % (idx, len(data_loader)))
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
        """The lmdb can not be pickled which creates problem when using DistributedDataParallel.
        Instead we discard it and reload it when we initialze the clone of the dataset
        """

        attributes = self.__dict__.copy()
        del attributes['env']
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        # NOTE: We need to reconnect to the database after unpickling.
        # This can create problems if using DistributedDataParallel
        # over mulitple machines.
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
