from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer

from rl_tracking.models.test import SimpleNN
from rl_tracking.preprocessing.hit_candidates import EventProcessor
#from rl_tracking.utils.loading import TrackingDataModule
from rl_tracking.utils.stream_loading import TrackingDataModule
data_dir = Path("/Users/liv/trackML/train_100_events/")
# Initialize dataset and dataloader
ep = EventProcessor(data_dir, 3)
dm = TrackingDataModule(file_paths = data_dir,
                        batch_size=64,
                        event_processor=ep)


dm.setup(stage="fit")
checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min')
model = SimpleNN()
trainer = Trainer(
    max_epochs=200,
    callbacks=[checkpoint_callback],
    accelerator='auto',  # Use GPU if available, otherwise CPU
    log_every_n_steps=10,
)


if __name__ == "__main__":
# Train the model
    trainer.fit(model, datamodule=dm)
