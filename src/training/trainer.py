import os
import math
import torch
import torch.nn.functional as F
import pickle
import datetime
from torch.amp import autocast, GradScaler

def train_decision_transformer(
    model, train_loader, val_loader,
    optimizer, device, num_epochs=50,
    checkpoint_dir="checkpoints"
):
    # Crear carpeta si no existe
    os.makedirs(checkpoint_dir, exist_ok=True)

    model.to(device)
    # Enable CuDNN autotuner for potentially better performance on fixed-size inputs
    torch.backends.cudnn.benchmark = True

    # Set model to training mode initially
    model.train()

    history = {'train_loss': [], 'val_loss': []}

    # AMP scaler for mixed-precision training
    scaler = GradScaler()

    # Guardar el mejor modelo según validación
    best_val_loss = float("inf")

    # Warmup/ scheduler configuration (warmup measured in epochs)
    warmup_epochs = 10
    # final learning rate is taken from optimizer's initial lr
    final_lr = optimizer.param_groups[0].get('lr', 1e-4)

    # Create the per-step scheduler (we need len(train_loader))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = warmup_epochs * steps_per_epoch

    def lr_lambda(current_step: int):
        if current_step < warmup_steps and warmup_steps > 0:
            return float(current_step) / float(max(1, warmup_steps))
        # cosine decay from 1 -> 0
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


    for epoch in range(num_epochs):

        # -------- TRAIN MODE ----------
        total_train_loss = 0

        for batch in train_loader:

            # Move data to device
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rtg = batch['rtg'].to(device)
            timesteps = batch['timesteps'].to(device)
            groups = batch['groups'].to(device)
            targets = batch['targets'].to(device)

            # Mixed precision forward
            with autocast('cuda'):
                logits = model(states, actions, rtg, timesteps, groups)
                # Compute loss (cross-entropy expects (B, C, L))
                loss = F.cross_entropy(
                    logits.transpose(1, 2),
                    targets,
                    ignore_index=-1
                )

            # Zero gradients
            optimizer.zero_grad()

            # Scale loss, backward, unscale, clip, step, update scaler
            scaler.scale(loss).backward()
            # Unscale before clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optimizer)
            scaler.update()

            # Step the LR scheduler per optimizer step (if created)
            if 'scheduler' in locals():
                scheduler.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        if epoch % 10 == 0:
            # -------- VALIDATION MODE ----------
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch in val_loader:
                    states = batch['states'].to(device)
                    actions = batch['actions'].to(device)
                    rtg = batch['rtg'].to(device)
                    timesteps = batch['timesteps'].to(device)
                    groups = batch['groups'].to(device)
                    targets = batch['targets'].to(device)

                    # Use autocast in validation for faster inference
                    with autocast('cuda'):
                        logits = model(states, actions, rtg, timesteps, groups)
                        loss = F.cross_entropy(
                            logits.transpose(1, 2),
                            targets,
                            ignore_index=-1
                        )

                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)

            # -------- CHECKPOINTING ----------
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss

                checkpoint_path = os.path.join(
                    checkpoint_dir,
                    f"dt_epoch{epoch+1}_val{avg_val_loss:.4f}.pt"
                )

                # Save checkpoint including scaler state for AMP
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "val_loss": avg_val_loss,
                    "train_loss": avg_train_loss,
                }, checkpoint_path)

                print(f"  ✔ Checkpoint guardado: {checkpoint_path}")

            # -------- LOG ----------
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val   Loss: {avg_val_loss:.4f}")

            
    # Guardamos el modelo entrenado
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    torch.save(model.state_dict(), f'trained_model_{date}.pt')
    print(f"Model saved to 'trained_model_{date}.pt'")

    # Guardamos el historial de entrenamiento
    with open(f'training_history_{date}.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"Training history saved to 'training_history_{date}.pkl'")
    
    return model, history
