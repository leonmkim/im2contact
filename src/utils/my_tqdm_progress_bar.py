from pytorch_lightning.callbacks import TQDMProgressBar

# overwrite the validation loop to stop showing all metrics
class MyTQDMProgressBar(TQDMProgressBar):
    
    def on_validation_end(self, trainer, pl_module):
        # don't show all metrics
        # super().on_validation_epoch_end(trainer, pl_module)
        # show only the loss
        if self.main_progress_bar is not None and trainer.state.fn == "fit":
            self.main_progress_bar.set_postfix({'val_loss': trainer.callback_metrics['val_loss']})
        self.val_progress_bar.close()
        self.reset_dataloader_idx_tracker()
