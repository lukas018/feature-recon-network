from tqdm import tqdm
import learn2learn as l2l
from learn2learn.data.transforms import RandomNWays, RandomKShots, LoadData, ConsecutiveLabels, RemapLabels
from learn2learn.data import MetaDataset, TaskDataset
from foi_fewshot.utils import initialize_taskdataset, split_dataset, fewshot_episode
from foi_fewshot.trainers import TrainingArguments, FewshotArguments, MetabatchWrapper
from torch.nn.parallel import DataParallel, DistributedDataParallel

from typing import Dict

ds = l2l.vision.datasets.MiniImagenet(root="~/Downloads", mode='train', download=True)
# nways = (2, 10)
# kquery = 10
# kways =  (kquery + 1, kquery+5)
nways = 5
kways = 5

total = 100
dl = initialize_taskdataset(ds, nways, kways, num_tasks=total, num_workers=1, batch_size=4)
lens = []
it = iter(dl)

# pb = tqdm(it, total=total)
# for batch in pb:
#     pb.set_description(f"bsz: {batch[0][0].shape}")


# print(imgs.shape)
# print(img.flatten(0,1).shape)

from foi_fewshot.models import ResNet12
from foi_fewshot.algorithms.meta_baseline import MetaBaseline
model = ResNet12()
mb = MetaBaseline(model)
# mb.init_pretraining(640, 64)
# training_args = TrainingArguments(
#     modeldir="~/Downloads/models",
#     logdir="~/Downloads/logs",
#     do_eval=True,
#     batch_size=4,
#     num_workers=4,
#     max_epochs=10,
# )

# pt = PreTrainer(
#     model,
#     ds,
#     training_args,
# )

# pt.train()

x = next(it)[0]
res = fewshot_episode(mb, x, 3)
args = FewshotArguments(
    modeldir=None,
    logdir=None,
    nways=nways,
    ksupport=2,
    kquery=3,
)
mw = MetabatchWrapper(mb, args)
dp = DataParallel(mw)
res = mw(next(it))
breakpoint()
res = dp(next(it))
breakpoint()
pass
