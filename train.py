import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from data_loader import GoProDataset
from model import UNet

def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for blur_images, sharp_images in tqdm(loader):
        blur_images = blur_images.to(device)
        sharp_images = sharp_images.to(device)

        optimizer.zero_grad()
        outputs = model(blur_images)
        loss = criterion(outputs, sharp_images)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * blur_images.size(0)
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for blur_images, sharp_images in tqdm(loader):
            blur_images = blur_images.to(device)
            sharp_images = sharp_images.to(device)

            outputs = model(blur_images)
            loss = criterion(outputs, sharp_images)

            running_loss += loss.item() * blur_images.size(0)
    return running_loss / len(loader.dataset)

def main():
    # Veri Dönüşümleri
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Veri Setleri ve Yükleyicileri
    train_dataset = GoProDataset(root_dir='GoPro_Large/train', transform=transform)
    test_dataset = GoProDataset(root_dir='GoPro_Large/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Cihaz Ayarı
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model, Kayıp Fonksiyonu ve Optimizasyon
    model = UNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20

    for epoch in range(num_epochs):
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate(model, test_loader, criterion, device)
        print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Modelin Kaydedilmesi
    torch.save(model.state_dict(), 'deblur_unet.pth')

if __name__ == '__main__':
    main()
