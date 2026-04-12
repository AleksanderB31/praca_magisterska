import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
from models.tcdcn import TCDCN
from utils.data_loader import MTFLDatasetRefined
import os

def calculate_accuracy(output, target):
    _, predicted = torch.max(output, 1)
    correct = (predicted == target).sum().item()
    return correct / target.size(0)

def validate(model, dataloader, criterion_landmarks, criterion_cls, criterion_glasses, device):
    model.eval()
    val_loss = 0.0
    val_mse_lm = 0.0
    val_acc_gender = 0.0
    val_acc_glasses = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            gender = batch['gender'].to(device)
            glasses = batch['glasses'].to(device)
            pose = batch['pose'].to(device)
            smile = batch['smile'].to(device)
            
            pred_landmarks, pred_gender, pred_glasses, pred_pose, pred_smile = model(images)
            
            loss_lm = criterion_landmarks(pred_landmarks, landmarks)
            loss_gender = criterion_cls(pred_gender, gender)
            loss_glasses = criterion_glasses(pred_glasses, glasses)
            loss_pose = criterion_cls(pred_pose, pose)
            loss_smile = criterion_cls(pred_smile, smile)
            
            loss = 100.0 * loss_lm + (loss_gender + loss_glasses + loss_pose + loss_smile)
            
            val_loss += loss.item()
            val_mse_lm += loss_lm.item()
            val_acc_gender += calculate_accuracy(pred_gender, gender)
            val_acc_glasses += calculate_accuracy(pred_glasses, glasses)
            
    n = len(dataloader)
    return val_loss/n, val_mse_lm/n, val_acc_gender/n, val_acc_glasses/n

def save_plots(history, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # Loss Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'loss_plot.png'))
    plt.close()
    
    # Accuracy Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_acc_gender'], label='Val Gender Acc')
    plt.plot(history['val_acc_glasses'], label='Val Glasses Acc')
    plt.title('Validation Accuracies')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'accuracy_plot.png'))
    plt.close()
    
    # Landmark MSE Plot
    plt.figure(figsize=(10, 5))
    plt.plot(history['val_mse_lm'], label='Val Landmark MSE')
    plt.title('Validation Landmark MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'landmark_mse_plot.png'))
    plt.close()

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    EPOCHS = 50
    NUM_LANDMARKS = 5
    
    # Paths
    DATASET_ROOT = "/home/aleksander/workspace/praca_magisterska/datasets/MTFL"
    TRAIN_TXT = os.path.join(DATASET_ROOT, "training.txt")
    
    if not os.path.exists(TRAIN_TXT):
        if os.path.exists(os.path.join(DATASET_ROOT, "MTFL", "training.txt")):
             DATASET_ROOT = os.path.join(DATASET_ROOT, "MTFL")
             TRAIN_TXT = os.path.join(DATASET_ROOT, "training.txt")
        else:
            print(f"Dataset not found at {TRAIN_TXT}.")
            return

    # Data Loader & Split
    full_dataset = MTFLDatasetRefined(root_dir=DATASET_ROOT, txt_file=TRAIN_TXT, augment=True)
    train_size = int(0.7 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.augment = False 
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Dataset split: Train={len(train_dataset)}, Val={len(val_dataset)}")

    # Model
    model = TCDCN(num_landmarks=NUM_LANDMARKS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    
    # Loss Functions
    # Weights for imbalanced attributes (Glasses: 1408 vs 8592)
    # Weight = 1 / count
    w_glasses = torch.tensor([1.0/1408, 1.0/8592]).to(device)
    w_glasses = w_glasses / w_glasses.sum() * 2.0 # Normalize
    
    criterion_landmarks = nn.MSELoss()
    criterion_cls = nn.CrossEntropyLoss()
    criterion_glasses = nn.CrossEntropyLoss(weight=w_glasses)
    
    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_mse_lm': [],
        'val_acc_gender': [],
        'val_acc_glasses': []
    }
    
    # Training Loop
    for epoch in range(EPOCHS):
        model.train()
        train_dataset.dataset.augment = True
        running_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].to(device)
            gender = batch['gender'].to(device)
            glasses = batch['glasses'].to(device)
            pose = batch['pose'].to(device)
            smile = batch['smile'].to(device)
            
            optimizer.zero_grad()
            pred_landmarks, pred_gender, pred_glasses, pred_pose, pred_smile = model(images)
            
            loss_lm = criterion_landmarks(pred_landmarks, landmarks)
            loss_gender = criterion_cls(pred_gender, gender)
            loss_glasses = criterion_glasses(pred_glasses, glasses)
            loss_pose = criterion_cls(pred_pose, pose)
            loss_smile = criterion_cls(pred_smile, smile)
            
            loss = 100.0 * loss_lm + (loss_gender + loss_glasses + loss_pose + loss_smile)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        # Validation Phase
        val_dataset.dataset.augment = False
        v_loss, v_mse, v_acc_gen, v_acc_glass = validate(model, val_loader, criterion_landmarks, criterion_cls, criterion_glasses, device)
        
        # Epoch Summary
        avg_train_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f}, Val Loss: {v_loss:.4f}, Val Gen Acc: {v_acc_gen:.2f}")
        
        # Update History
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(v_loss)
        history['val_mse_lm'].append(v_mse)
        history['val_acc_gender'].append(v_acc_gen)
        history['val_acc_glasses'].append(v_acc_glass)
        
        scheduler.step(v_loss)
        
        # Save best model
        os.makedirs("research-on-real/models/weights", exist_ok=True)
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            torch.save(model.state_dict(), "research-on-real/models/weights/tcdcn_best.pth")
            print(f"--> Best model saved at epoch {epoch+1}")
            
        torch.save(model.state_dict(), f"research-on-real/models/weights/tcdcn_epoch_{epoch+1}.pth")

    # Save Plots
    save_plots(history, 'research-on-real/plots')
    print("Training finished. Plots saved to research-on-real/plots/")

if __name__ == "__main__":
    train()
