from monai.apps import DecathlonDataset

root_dir = "../data"

train_ds = DecathlonDataset(
    root_dir=root_dir,
    task='Task09_Spleen',
    section="training",
    cache_rate=0,
    num_workers=4,
    download=True,
    seed=0
)