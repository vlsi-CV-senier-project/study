import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import torch.quantization

from thop import profile
from tqdm import tqdm

# 명령줄 인자 설정
parser = argparse.ArgumentParser(description="PyTorch CIFAR-100 Training with Pruning and Quantization")
parser.add_argument("--epochs", default=50, type=int, help="number of total epochs to run")
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--batch-size", default=128, type=int, help="mini-batch size")
parser.add_argument("--prune-method", default="magnitude", choices=["magnitude", "structured", "channel"], help="pruning method")
parser.add_argument("--prune-rate", default=0.1, type=float, help="pruning rate")
parser.add_argument("--quantization-type", default="dynamic", choices=["dynamic", "qat"], help="quantization type")
parser.add_argument("--quantization-bits", default=8, type=int, choices=[4, 8], help="quantization bits")
parser.add_argument("--qat-epochs", default=5, type=int, help="number of epochs for quantization-aware training")


# CIFAR-100 데이터셋 로드
def load_data(batch_size):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader

# train 함수 정의
def train(model, device, train_loader, optimizer, epoch, criterion, args):  
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for batch_idx, (data, target) in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
        loop.set_postfix(loss=loss.item())

def train_cpu(model, device, train_loader, optimizer, epoch, criterion, args):  
    model.train()
    loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for batch_idx, (data, target) in loop:
        data, target = data.to('cpu'), target.to('cpu')
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        loop.set_description(f"Epoch [{epoch}/{args.epochs}]")
        loop.set_postfix(loss=loss.item())

# 평가 함수
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

def test_cpu(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to('cpu'), target.to('cpu')
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(f"\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n")

# 가지치기 및 양자화 적용 함수 (이전 단계에서 정의한 apply_pruning 및 apply_quantization 함수 포함)

def count_zero_weights(model):
    total_weights = 0
    zero_weights = 0
    for param in model.parameters():
        total_weights += param.numel()
        zero_weights += (param == 0).sum().item()
    zero_weight_percentage = 100.0 * zero_weights / total_weights
    return total_weights, zero_weights, zero_weight_percentage

def main():
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 100)  # CIFAR-100 클래스 수에 맞게 조정
    model.to(device)  

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    train_loader, test_loader = load_data(args.batch_size)
    
    input = torch.randn(1, 3, 224, 224).to(device)
    flops_before, params_before = profile(model, inputs=(input,), verbose=False)
    total_weights_before, zero_weights_before, zero_weight_percentage_before = count_zero_weights(model)
    print(f"Before pruning - Model FLOPs: {flops_before}, Parameters: {params_before}, Zero weight percentage: {zero_weight_percentage_before:.2f}%")
    
    # 가지치기 적용
    apply_pruning(model, args.prune_method, args.prune_rate)

    apply_custom_conv_layers(model)
    
    flops_after, params_after = profile(model, inputs=(input,), verbose=False)
    total_weights_after, zero_weights_after, zero_weight_percentage_after = count_zero_weights(model)
    print(f"After pruning - Model FLOPs: {flops_after}, Parameters: {params_after}, Zero weight percentage: {zero_weight_percentage_after:.2f}%")
    
    # main 함수 내에서 train 함수 호출 부분
    for epoch in range(1, args.epochs + 1):

        train(model, device, train_loader, optimizer, epoch, criterion, args)  # args 추가
        """
        flops_train, params_train = profile(model, inputs=(input,), verbose=False)
        total_weights_train, zero_weights_train, zero_weight_percentage_train = count_zero_weights(model)
        print(f"After training - Model FLOPs: {flops_train}, Parameters: {params_train}, Zero weight percentage: {zero_weight_percentage_train:.2f}%")
        """
        test(model, device, test_loader, criterion)
        scheduler.step()


    model = apply_quantization(model, args.quantization_type, args.quantization_bits, device, train_loader, criterion, optimizer, args.qat_epochs, args)
    # 양자화 및 가지치기가 적용된 모델로 평가를 수행합니다.

    test_cpu(model, device, test_loader, criterion)

    # thop를 사용하여 FLOPs와 매개변수 수 계산
    input = torch.randn(1, 3, 224, 224).to('cpu')
    flops_quantized, params_quantized = profile(model, inputs=(input,), verbose=False)
    total_weights_quantized, zero_weights_quantized, zero_weight_percentage_quantized = count_zero_weights(model)
    print(f"After quantization - Model FLOPs: {flops_quantized}, Parameters: {params_quantized}, Zero weight percentage: {zero_weight_percentage_quantized:.2f}%")



def apply_pruning(model, prune_method, prune_rate):
   
    parameters_to_prune = []
    
    # 모든 Conv2d 및 Linear 레이어를 대상으로 가지치기 대상을 선택합니다.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    # 선택한 가지치기 방법에 따라 가지치기를 적용합니다.
    if prune_method == "magnitude":
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=prune_rate,
        )
    elif prune_method == "structured":
        for module, _ in parameters_to_prune:
            prune.ln_structured(module, name="weight", amount=prune_rate, n=2, dim=0)
    elif prune_method == "channel":
        for module, _ in parameters_to_prune:
            prune.ln_structured(module, name="weight", amount=prune_rate, n=2, dim=1) # Conv2d의 경우 채널 가지치기


    # 가지치기 적용 후, 가지치기 마스크를 영구적으로 만듭니다.
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')


def apply_quantization(model, quantization_type, num_bits, device, train_loader, criterion, optimizer, qat_epochs, args):

    model.to('cpu')  # 양자화는 CPU에서 수행됩니다.
    device = torch.device('cpu')

    if quantization_type == 'dynamic':
        # Dynamic Quantization
        if num_bits == 8:
            model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        elif num_bits == 4:
            model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

            for name, module in model_quantized.named_modules():
                if hasattr(module, 'weight'):
                    weight = module.weight()
                    new_scale = weight.q_scale() * 2  # 스케일을 조정하여 값의 범위를 4비트에 맞춤
                    new_zero_point = max(min(weight.q_zero_point(), 15), -16)
                    # 새로운 스케일과 제로 포인트로 가중치를 다시 양자화
                    module.weight().data = torch.quantize_per_tensor(weight.dequantize(), new_scale, new_zero_point, torch.qint8)


    elif quantization_type == 'qat':
        # Quantization-Aware Training (QAT)
        model.train()
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=False)

        model.to('cuda')
        device = torch.device('cuda')
        for epoch in range(qat_epochs):
            train(model, device, train_loader, optimizer, epoch, criterion, args)

        if num_bits == 8:
            # QAT 후 8-bit 양자화를 적용
            device = torch.device('cpu')
            model.to('cpu')
            model.eval()
            model_quantized = torch.quantization.convert(model.eval(), inplace=True)
        elif num_bits == 4:
            # QAT 후 모델을 8비트로 양자화하고, 4비트처럼 조정
            model_quantized = torch.quantization.convert(model.eval(), inplace=False)
            for name, module in model_quantized.named_modules():
                if hasattr(module, 'weight'):
                    weight = module.weight()
                    new_scale = weight.q_scale() * 2  # 스케일 조정
                    new_zero_point = max(min(weight.q_zero_point(), 15), -16)
                    module.weight().data = torch.quantize_per_tensor(weight.dequantize(), new_scale, new_zero_point, torch.qint8)

    model_quantized.to('cpu')
    return model_quantized

class CustomConv2d(nn.Module):
    def __init__(self, conv, mask):
        super(CustomConv2d, self).__init__()
        self.conv = conv
        self.mask = mask.to(conv.weight.device)  # 마스크를 가중치와 동일한 디바이스로 이동
        self.weight = nn.Parameter(conv.weight.data.clone())

        # 마스크 적용하여 가중치 초기화
        self.weight.data.mul_(self.mask)

        if conv.bias is not None:
            self.bias = nn.Parameter(conv.bias.data)
        else:
            self.register_parameter('bias', None)

        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups

    def forward(self, x):
        # self.weight와 동일한 디바이스로 self.mask 이동
        if self.mask.device != self.weight.device:
            self.mask = self.mask.to(self.weight.device)
            
        # 0이 아닌 가중치만을 사용하여 연산을 수행합니다.
        weight = self.weight * self.mask
        return F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def apply_custom_conv_layers(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Conv2d):
            # 가지치기 마스크 가져오기
            mask = module.weight.data != 0  # 가중치가 0이 아니면 True, 0이면 False
            custom_conv = CustomConv2d(module, mask)
            setattr(model, name, custom_conv)
        else:
            apply_custom_conv_layers(module)

if __name__ == '__main__':
    main()
