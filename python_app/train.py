import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import model

# Konfigurace
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'czech-coins')
MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'coin_model.pth')
BATCH_SIZE = 32
EPOCHS = 40  # Více epoch pro lepší konvergenci
LEARNING_RATE = 0.0001
IMG_SIZE = (224, 224)

def train():
    # 1. Kontrola existence dat
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory '{DATA_DIR}' not found.")
        return

    print(f"Loading data from {DATA_DIR}...")

    # 2. Transformace dat
    # ResNet vyžaduje normalizaci se statistikami ImageNet
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE), # ResNet standard 224x224 pro lepsi detaily
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Augmentace pro robustnost
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180), # Mince jsou invariantní vůči rotaci
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        # Agresivní změna jasu/kontrastu pro simulaci špatného osvětlení
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.05),
        # KLÍČOVÉ: Vysoká pravděpodobnost grayscale nutí model učit se tvar (ražbu), ne barvu
        transforms.RandomGrayscale(p=0.5),
    ])

    # 3. Načtení datasetu
    full_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    print(f"Classes found: {full_dataset.classes}")
    
    # Rozdělení Train/Val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"Training on {len(train_dataset)} images, Validating on {len(val_dataset)} images.")

    # 4. Inicializace modelu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Použití modelu ResNet
    net = model.get_coin_model(num_classes=len(full_dataset.classes)).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

    # 5. Trénovací smyčka
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")

        # Validace
        net.eval()
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        print(f"Validation Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), MODEL_SAVE_PATH)
            print("Model saved!")

    print("Training Finished.")
    print(f"Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to {os.path.abspath(MODEL_SAVE_PATH)}")

if __name__ == "__main__":
    train()
