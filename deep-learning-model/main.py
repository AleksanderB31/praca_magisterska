import torch
from torch.utils.data import DataLoader
from driver_behavior.dataset import DriverBehaviorDataset, get_transforms
from driver_behavior.model import TinyVGG
from driver_behavior.train import train, test_step
from driver_behavior.utils import save_plots, save_confusion_matrix, save_metrics
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='Driver Behavior Detection Training')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setup directories
    data_path = "../datasets/deep-learning-dataset"
    train_dir = os.path.join(data_path, "train")
    test_dir = os.path.join(data_path, "test")
    
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create datasets and dataloaders
    train_dataset = DriverBehaviorDataset(
        root_dir=train_dir, 
        transform=get_transforms(is_train=True),
        is_train=True
    )
    test_dataset = DriverBehaviorDataset(
        root_dir=test_dir, 
        transform=get_transforms(is_train=False),
        is_train=False
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=os.cpu_count())
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=os.cpu_count())

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Initialize model
    model = TinyVGG(
        input_shape=3, # RGB
        hidden_units=10, 
        output_shape=len(train_dataset.classes)
    ).to(device)

    # Loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    print("Starting training...")
    results = train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=args.epochs,
        device=device,
        save_dir=results_dir
    )

    # Save plots
    save_plots(
        results['train_acc'], 
        results['train_loss'], 
        results['test_acc'], 
        results['test_loss'], 
        results_dir
    )

    # Final Evaluation
    print("Running final evaluation...")
    model.load_state_dict(torch.load(os.path.join(results_dir, "best_model.pth")))
    _, _, y_true, y_pred = test_step(model, test_dataloader, loss_fn, device)
    
    # Save metrics and confusion matrix
    save_confusion_matrix(y_true, y_pred, train_dataset.classes, results_dir)
    acc, f1 = save_metrics(y_true, y_pred, train_dataset.classes, results_dir)
    
    print(f"Final Accuracy: {acc:.4f}")
    print(f"Final F1 Score: {f1:.4f}")
    print(f"Results saved to {results_dir}")

if __name__ == '__main__':
    main()
