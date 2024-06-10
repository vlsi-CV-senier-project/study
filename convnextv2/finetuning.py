import torch
from torch import nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader
import argparse
from fcmae import convnextv2_tiny

def get_args_parser():
    parser = argparse.ArgumentParser('FCMAE Model Utilization', add_help=False)
    parser.add_argument('--model_path', type=str, default='./best_model_epoch_32.pth', help='Path to the pre-trained FCMAE model')
    parser.add_argument('--data_path', type=str, default='./data', help='Path to the dataset')
    parser.add_argument('--device', default='cuda', help='Device to use for inference/training')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')
    parser.add_argument('--img_size', type=int, default=32, help='Image size for transformations')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for optimizer')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train')
    return parser

def load_pretrained_model(model_path, device):
    model = convnextv2_tiny()
    print(model)  # Print model architecture to confirm it is as expected
    checkpoint = torch.load(model_path, map_location=device)
    model_dict = model.state_dict()
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in model_dict}
    model_dict.update(filtered_checkpoint)
    model.load_state_dict(model_dict)
    model.to(device)
    return model

def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = load_pretrained_model(args.model_path, device)
    test_transforms = Compose([
        Resize((args.img_size, args.img_size)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=test_transforms)
    testloader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = outputs[0] if isinstance(outputs, tuple) else outputs

            print(f"Output shape after correction: {outputs.shape}")  # Further debugging
            if outputs.nelement() == 0:
                raise ValueError("The output tensor is empty. Check model's forward method.")

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the {len(testset)} test images: {accuracy:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('FCMAE Model Utilization', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
