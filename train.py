import os
import torch
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, load_checkpoint, print_examples
from get_loader import get_loader
from model import CNNtoRNN


def train():
    transform = transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_loader, dataset = get_loader(
        root_folder="D:/Image_Captioning/flickr8k/images",
        annotation_file="D:/Image_Captioning/flickr8k/captions.txt",
        transform=transform,
        num_workers=2,
    )

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_model = True
    save_model = True
    train_CNN = False

    # Hyperparameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(dataset.vocab)
    num_layers = 2
    learning_rate = 3e-4
    num_epochs = 20
    dropout = 0.3
    batch_size = 32

    # Early stopping
    early_stop_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # TensorBoard
    writer = SummaryWriter("runs/flickr")
    step = 0

    # Model, loss, optimizer
    model = CNNtoRNN(embed_size, hidden_size, vocab_size, num_layers, dropout).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Only finetune the CNN
    for name, param in model.encoderCNN.inception.named_parameters():
        if "fc.weight" in name or "fc.bias" in name:
            param.requires_grad = True
        else:
            param.requires_grad = train_CNN

    # Checkpoint setup
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pth.tar")

    # Load checkpoint safely
    if load_model and os.path.exists(checkpoint_path):
        print("Found checkpoint, attempting to load...")
        checkpoint = torch.load(checkpoint_path)

        # Load model weights, ignore missing/mismatched keys
        try:
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            print("Model weights loaded (partial if mismatch).")
        except Exception as e:
            print(f"Could not load model weights: {e}")

        # Try loading optimizer state only if shapes match
        try:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer state loaded.")
        except Exception as e:
            print(f"Could not load optimizer state (probably changed layer sizes): {e}")
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            print("Optimizer reinitialized.")

        step = checkpoint.get("step", 0)
    else:
        print("No checkpoint found, starting fresh!")
        step = 0

    model.train()

    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch+1}/{num_epochs}")
        print_examples(model, device, dataset)
        print("\n")

        loop = tqdm(train_loader, total=len(train_loader), leave=False)
        epoch_loss = 0.0

        for idx, (imgs, captions) in enumerate(loop):
            imgs = imgs.to(device)
            captions = captions.to(device)

            outputs = model(imgs, captions[:-1])
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )

            writer.add_scalar("Training loss", loss.item(), global_step=step)
            step += 1
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")
            loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")

        # Early stopping check
        val_loss = avg_epoch_loss  # Replace with real validation loss if available
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save best model checkpoint
            best_path = os.path.join(checkpoint_dir, "best_model.pth.tar")
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, filename=best_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save checkpoint for this epoch
        if save_model:
            epoch_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pth.tar")
            latest_path = os.path.join(checkpoint_dir, "latest.pth.tar")
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, filename=latest_path)
            save_checkpoint({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step
            }, filename=epoch_path)
            print(f"Saved checkpoint for epoch {epoch+1}!")

    writer.flush()


if __name__ == "__main__":
    train()
