import torch
from tqdm.notebook import tqdm


def train_model(model, dataloaders, criterion, optimizer, scheduler=None, num_epochs=50, device='cuda', name='baseline', disable=False):
    train_loss_history = []
    val_loss_history = []

    best_loss = 1e10

    save_file = f'{name}_best.pt'

    for _ in tqdm(range(1, num_epochs + 1), disable=disable):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if scheduler is not None:
                scheduler.step()

            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), save_file)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            elif phase == 'test':
                val_loss_history.append(epoch_loss)

    model.load_state_dict(torch.load(save_file))
    return model, train_loss_history, val_loss_history


def train_inalu_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=50, device='cuda', name='inalu'):
    train_loss_history = []
    val_loss_history = []

    best_loss = 1e10
    good_epoch = 1

    save_file = f'{name}_best.pt'

    for epoch in tqdm(range(1, num_epochs + 1)):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    running_loss += loss.item() * inputs.size(0)

                    if phase == 'train' and epoch > 10:
                        loss += model.reg_loss()

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            if scheduler is not None:
                scheduler.step()

            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                good_epoch = epoch
                torch.save(model.state_dict(), save_file)

            if phase == 'test' and epoch - good_epoch > 10:
                model.reinitialize()
                model.to(device)

            if phase == 'train':
                train_loss_history.append(epoch_loss)
            elif phase == 'test':
                val_loss_history.append(epoch_loss)

    model.load_state_dict(torch.load(save_file))
    return model, train_loss_history, val_loss_history
