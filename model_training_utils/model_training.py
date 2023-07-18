import torch
import numpy as np
import time
 
def checkpoint(model, optimizer, filename):
    states = {
        'optimizer_state': optimizer.state_dict(),
        'model_state': model.state_dict(),
    }
    torch.save(states, filename)
    
def resume(model, optimizer, filename):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])

    
def lr_decay(optimizer, epoch):
    if epoch%30==0:
        new_lr = learning_rate / (10**(epoch//10))
        optimizer = setlr(optimizer, new_lr)
        print(f'Changed learning rate to {new_lr}')
    return optimizer

def get_device():
    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')
    return device

def training(model, optimizer, loss_fn, train_dataloader, val_dataloader, model_path:str, epochs:int=200, early_stop_thresh:int=10, lr_decay=None, device=get_device(), start_epoch:int=0):
    best_val_accuracy = -1
    best_epoch = -1
    train_losses = []
    val_losses = []
    
    if start_epoch > 0:
        resume(model, optimizer, f"{model_path}/epoch_{start_epoch}_states.pth")
        model.to(device)
        start_epoch += 1
    
    for epoch in range(start_epoch, epochs):
        if lr_decay is not None:
            optimizer = lr_decay(optimizer, epoch)
            
        model.train()
        
        train_batch_losses=[]
        train_labels=[]
        train_predictions=[]
        
        training_start_time = time.time()
        
        # training loop
        for x_batch, y_batch in train_dataloader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_batch_losses.append(loss.item())
            train_labels.append(y_batch.detach().cpu().numpy())
            train_predictions.append(y_pred.detach().cpu().numpy().argmax(axis=1))  
        
        training_end_time = time.time()
        
        model.eval()
        
        train_losses.append(np.mean(train_batch_losses))
        train_labels = np.concatenate(train_labels)
        train_predictions = np.concatenate(train_predictions)
        train_accuracy = np.mean(train_labels==train_predictions)
        train_accuracy = float(train_accuracy) * 100
        print(f"End of epoch {epoch}: training accuracy = {train_accuracy:.2f}%, training loss = {train_losses[-1]}, training time taken = {(training_end_time - training_start_time):.2f} seconds")
        
        val_batch_losses=[]
        val_labels=[]
        val_predictions=[]
        
        val_start_time = time.time()
        
        # validation loop
        for x_batch, y_batch in val_dataloader:
            x_batch = x_batch.to(device, dtype=torch.float32)
            y_batch = y_batch.to(device, dtype=torch.long)
            
            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)
            
            val_batch_losses.append(loss.item())
            val_labels.append(y_batch.detach().cpu().numpy())
            val_predictions.append(y_pred.detach().cpu().numpy().argmax(axis=1))  
        
        val_end_time = time.time()
        
        val_losses.append(np.mean(val_batch_losses))
        val_labels = np.concatenate(val_labels)
        val_predictions = np.concatenate(val_predictions)
        val_accuracy = np.mean(val_labels==val_predictions)
        val_accuracy = float(val_accuracy) * 100
        print(f"End of epoch {epoch}: validation accuracy = {val_accuracy:.2f}%, validation loss = {val_losses[-1]}, validation time taken = {(val_end_time - val_start_time):.2f} seconds")
        
        # early stopping logic
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
        elif epoch - best_epoch > early_stop_thresh:
            print("Early stopped training at epoch %d" % epoch)
            resume(model, optimizer, f"{model_path}/epoch_{best_epoch}_states.pth")
            break  # terminate the training loop
        
        checkpoint(model, optimizer, f"{model_path}/epoch_{epoch}_states.pth")
    
    return train_losses, val_losses
 
