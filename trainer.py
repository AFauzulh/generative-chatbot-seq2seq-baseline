import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import wandb

def train(model, num_epochs, lr, tokenizer, train_dataset, val_dataset, save_dir, RANDOM_SEED, device, opt, crit='CEL'):
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    pad_idx = tokenizer.word2index["<PAD>"]
    if crit == "CTC":
        criterion = nn.CTCLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_losses = []

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    best_val_loss = float('inf')
        
    if opt.wandb_log:
        wandb.init(
            project="tesis-chatbot-siet",
#                 config={
#                     "learning_rate": opt.lr,
#                     "epochs": opt.num_epochs,
#                     "batch_size": opt.batch_size,
#                     "embedding_dim": opt.embedding_dim,
#                     "hidden_size": opt.hidden_size,
#                     "dropout": opt.dropout,
#                 },
            entity='alfirsa-lab',
            name=f"{opt.exp_name}"
        )
    for epoch in range(num_epochs):
    # for _ in progress:
        # start = time.time()
        # print(f"Epoch [{epoch+1}/{num_epochs}]")

        num_batch = 0
        val_num_batch = 0
        batch_loss = 0
        val_batch_loss = 0

        training_time = 0

        model.train()
        for (batch_idx, (X_train, y_train, input_len, target_len)) in enumerate(bar := tqdm(train_dataset)):
            X, y, input_lengths, target_lengths = sort_within_batch(X_train, y_train, input_len, target_len)
            
            X = X.permute(1,0)
            y = y.permute(1,0)

            inp_data = X.to(device)
            target = y.to(device)
            # target shape = (target_length, batch_size))

            # print(inp_data.shape, target.shape)
            output = model(inp_data, target, input_lengths)
            # # output shape = (target_length, batch_size, output_dim)
            
            # print(output.shape, target.shape)

            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)
            # membuat output shape menjadi (target_length*batch_size, output dim) dan target shape menjadi (target_length*batch_size) untuk dipassing ke loss function

            # print(output.shape, target.shape)
            optimizer.zero_grad()
            loss = loss_function(real=target, pred=output, input_lengths=input_lengths, target_lengths=target_lengths, criterion=criterion, crit=crit)
            batch_loss += loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.grad_clip)
            optimizer.step()
            bar.set_description(f'Train Seq2Seq Model '
                                         f'[train_loss={loss:.4f}'
                                         )
            num_batch+=1
        
        # Validation Process
        with torch.no_grad():
           model.eval()
           for (batch_idx, (X_val, y_val, input_len, target_len)) in enumerate(val_dataset):
            X, y, input_lengths, target_lengths = sort_within_batch(X_val, y_val, input_len, target_len)
            
            X = X.permute(1,0)
            y = y.permute(1,0)

            inp_data = X.to(device)
            target = y.to(device)
            output = model(inp_data, target, input_lengths)
            
            output = output[1:].reshape(-1, output.shape[2])
            target = target[1:].reshape(-1)

            val_loss = loss_function(real=target, pred=output, input_lengths=input_lengths, target_lengths=target_lengths, criterion=criterion, crit=crit)
            val_batch_loss += val_loss
            
            val_num_batch+=1
            
        train_loss_ = batch_loss/num_batch
        train_losses.append(train_loss_)
        
        val_loss_ = val_batch_loss/val_num_batch
        val_losses.append(val_loss_)
        
        if val_loss_ < best_val_loss:
            torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/best_loss.pth')
            
        torch.save(model.state_dict(), f'./saved_models/{opt.exp_name}/model.pth')
        # train_time = time.time() - start
        # training_time += train_time
        
        scheduler.step(val_loss_)
        
        if opt.wandb_log:
            wandb.log({"loss": train_loss_, "val_loss": val_loss_})
        
        print(
                f'Epochs: {epoch + 1} | Train Loss: {train_loss_:.3f} \
                | Val Loss: {val_loss_:.3f}\n')
    
    if opt.wandb_log:
        wandb.finish()
# def loss_function(real, pred, criterion):
#     """ Only consider non-zero inputs in the loss; mask needed """
#     #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
#     #print(mask)
#     mask = real.ge(1).type(torch.cuda.FloatTensor)
    
#     loss_ = criterion(pred, real) * mask 
#     return torch.mean(loss_)

def loss_function(real, pred, criterion, input_lengths, target_lengths, crit="CEL"):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)
    mask = real.ge(1).type(torch.cuda.FloatTensor)
    if crit == "CTC" :
        loss_ = criterion(pred, real, input_lengths, target_lengths) * mask
    else :
        loss_ = criterion(pred, real) * mask 
    return torch.mean(loss_)

### sort batch function to be able to use with pad_packed_sequence
# def sort_within_batch(X, y, lengths):
#     lengths, indx = lengths.sort(dim=0, descending=True)
#     X = X[indx]
#     y = y[indx]
#     return X, y, lengths # transpose (batch x seq) to (seq x batch)

def sort_within_batch(X, y, lengths, trg_len):
    lengths, indx = lengths.sort(dim=0, descending=True)
    X = X[indx]
    y = y[indx]
    trg_len = trg_len[indx]
    return X, y, lengths, trg_len # transpose (batch x seq) to (seq x batch)
