import torch
from PIL import Image
from torchvision import transforms
import argparse

from model import UNet

def deblur_image(model, image_path, device):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_image)

    output_image = output.cpu().squeeze(0)
    output_image = transforms.ToPILImage()(output_image.clamp(0, 1))
    return output_image

def main():
    parser = argparse.ArgumentParser(description='Deblur an image using trained model.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input blurred image.')
    parser.add_argument('--output', type=str, required=True, help='Path to save the deblurred image.')
    parser.add_argument('--model', type=str, default='deblur_unet.pth', help='Path to the trained model.')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Modeli yükleyin
    model = UNet().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    # Görüntüyü netleştirin
    output_image = deblur_image(model, args.input, device)

    # Sonucu kaydedin
    output_image.save(args.output)
    print(f'Deblurred image saved to {args.output}')

if __name__ == '__main__':
    main()
