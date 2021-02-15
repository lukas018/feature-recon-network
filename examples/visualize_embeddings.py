from pathlib import Path
import torch
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import plotly.express as px

from foi_fewshot.algorithms import MetaBaseline
from foi_fewshot.data import mini_imagenet
from foi_fewshot.data import split_dataset
from foi_fewshot.models import ResNet12
from foi_fewshot.data.utils import initialize_taskloader

# Initialize the model
model = ResNet12()
meta_baseline = MetaBaseline(model)
meta_baseline.init_pretraining(640, 64)
_, _, ds_test = mini_imagenet("~/Downloads")

# Path to a trained model
PATH = "~/Downloads/fewshot-models/fewshot-training-6000/model.pkl"
# Load pretrained model
path_to_checkpoint = Path(PATH).expanduser()
meta_baseline.load_state_dict(torch.load(path_to_checkpoint))

# Create a single task loader 5way5shot with 20 query images
dl = initialize_taskloader(ds_test, 5, 50, 45, 1, 1, 1)
batch = next(iter(dl))

with torch.no_grad():
    # Compute the class centroids
    meta_baseline.compute_centroids(support=batch["support"][0], cache=True)
    # Do a forward pass with the inner model to get the embeddings
    query_embeddings = meta_baseline.model(batch["query"][0])

    # Get numpy
    ce = meta_baseline.cached_centroids.cpu().numpy()
    qe = query_embeddings.cpu().numpy()
    q_labels = batch["query_labels"][0].numpy()

# Visualize using t-sne
tsne = TSNE(n_components=2, random_state=42)
tsne_vectors = tsne.fit_transform(np.concatenate((ce, qe)))

tsne_ce = tsne_vectors[:5]
tsne_qe = tsne_vectors[5:]

tsne_records = [
    {
        "x": emb[0],
        "y": emb[1],
        "marker": 0,
        "color": cls,
        "text": f"centriod: {cls}",
        "size": 10,
    }
    for cls, emb in enumerate(tsne_ce)
]
tsne_records2 = [
    {
        "x": emb[0],
        "y": emb[1],
        "marker": 1,
        "color": cls,
        "text": f"query: {cls}",
        "size": 1,
    }
    for emb, cls in zip(tsne_qe, q_labels)
]
df = pd.DataFrame([*tsne_records, *tsne_records2])

fig = px.scatter(
    df,
    x="x",
    y="y",
    color="color",
    hover_data=["text"],
    symbol=df["marker"],
    size="size",
    color_discrete_sequence=["blue", "orange", "green", "red", "purple"],
)
fig.show()
