import torch


def save_checkpoint(epoch, model, optimizer, training_losses, scheduler=None, args=None, 
                    validation_losses=None, validation_contra_losses=None,
                    file_path=None, learning_rate_dynamics=None):
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
        'training_losses': training_losses,
        'validation_losses': validation_losses,
        'validation_contra_losses': validation_contra_losses,
        'learning_rate_dynamics': learning_rate_dynamics
    }
    torch.save(checkpoint, file_path)


class CheckpointLoader:
    def __init__(self, file_path):
        self.checkpoint = None
        self.start_epoch = None
        self.training_losses = None
        self.validation_losses = None
        self.validation_contra_losses = None
        self.learning_rate_dynamics = None
        self.args = None
        self.checkpoint = torch.load(file_path)

    def load_args(self):
        self.args = self.checkpoint['args']
        return self.args

    def load_checkpoint(self, model, optimizer, scheduler=None):
        model.load_state_dict(self.checkpoint['model_state_dict'])
        optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        if scheduler is not None:
            scheduler.load_state_dict(self.checkpoint['scheduler_state_dict'])

        return model, optimizer, scheduler

    def load_epoch(self):
        """ Return the epoch at which training was interrupted. """
        self.start_epoch = self.checkpoint['epoch']
        return self.start_epoch

    def load_training_losses(self):
        """ Return the list of recorded training losses. """
        self.training_losses = self.checkpoint.get('training_losses', [])
        return self.training_losses

    def load_validation_losses(self):
        """ Return the list of recorded validation losses. """
        self.validation_losses = self.checkpoint.get('validation_losses', [])
        return self.validation_losses