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