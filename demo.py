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
parser.add_argument("--epochs", default=5, type=int, help="number of total epochs to run")
parser.add_argument("--lr", "--learning-rate", default=1e-3, type=float, help="initial learning rate")
parser.add_argument("--batch-size", default=128, type=int, help="mini-batch size")
parser.add_argument("--prune-method", default="channel", choices=["magnitude", "structured", "channel"], help="pruning method")
parser.add_argument("--prune-rate", default=0.1, type=float, help="pruning rate")
parser.add_argument("--quantization-type", default="dynamic", choices=["dynamic", "qat"], help="quantization type")
parser.add_argument("--quantization-bits", default=8, type=int, choices=[4, 8], help="quantization bits")

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

# 훈련 함수
# train 함수 정의
def train(model, device, train_loader, optimizer, epoch, criterion, args):  # args 매개변수 추가
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

# 가지치기 및 양자화 적용 함수 (이전 단계에서 정의한 apply_pruning 및 apply_quantization 함수 포함)

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

    # thop를 사용하여 FLOPs와 매개변수 수 계산
    input = torch.randn(1, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print(f"Model FLOPs: {flops}, Parameters: {params}")

    # main 함수 내에서 train 함수 호출 부분
    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, criterion, args)  # args 추가
        test(model, device, test_loader, criterion)
        scheduler.step()


    # 여기에 가지치기 및 양자화 적용 코드 추가

# 가지치기 및 양자화 기능을 추가하기 위해 필요한 부분을 수정합니다.
# 가지치기 함수를 정의하고, 가지치기를 적용할 시점과 방법을 구체화합니다.

def apply_pruning(model, prune_method, prune_rate):
   
    parameters_to_prune = []
    
    # 모든 Conv2d 및 Linear 레이어를 대상으로 가지치기 대상을 선택합니다.
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
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
            if isinstance(module, torch.nn.Conv2d):
                prune.ln_structured(module, name="weight", amount=prune_rate, n=2, dim=1) # Conv2d의 경우 채널 가지치기
            else:
                prune.ln_structured(module, name="weight", amount=prune_rate, n=2, dim=0) # Linear 레이어는 축 선택

    # 가지치기 적용 후, 가지치기 마스크를 영구적으로 만듭니다.
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

# 가지치기 적용 부분을 main 함수 내에 포함합니다.
# 이 코드 블록은 직접 실행되지 않으며, 실제 환경에서의 실행을 위해서는 적절한 위치에 삽입 및 조정이 필요합니다.

# 예시로, 훈련 후 테스트 전에 가지치기를 적용할 수 있습니다.
# apply_pruning(model, args.prune_method, args.prune_rate)

# 가지치기가 적용된 모델로 평가를 수행합니다.
# test(model, device, test_loader, criterion)

# 양자화 로직은 아직 구현되지 않았습니다. 양자화에 대한 구체적인 요구 사항이나 선호도에 따라 추가 구현이 필요합니다.

# 주석 처리된 코드는 실행을 위한 지침을 제공하기 위함입니다. 실제 코드 내에서 적절한 위치에 배치하고 필요에 따라 수정해야 합니다.

# 양자화를 적용하는 함수를 정의합니다. 이 함수는 Dynamic Quantization과 Quantization-Aware Training(QAT)을 선택할 수 있게 하며,
# 8-bit 및 4-bit 양자화를 적용할 수 있도록 합니다. argparse를 사용하여 사용자가 이러한 옵션을 선택할 수 있도록 합니다.

def apply_quantization(model, quantization_type, num_bits, device):
    model.to('cpu')  # 양자화는 CPU에서 수행됩니다.
    
    if quantization_type == 'dynamic':
        # Dynamic Quantization
        if num_bits == 8:
            model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        elif num_bits == 4:
            model_quantized = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
            # 8비트 양자화 모델을 4비트로 조정하는 실험적 접근
            # 여기서는 단순히 스케일을 조정하는 방법을 사용합니다. 
            # 실제 양자화 알고리즘을 구현하려면 더 복잡한 로직이 필요합니다.
            for name, module in model_quantized.named_modules():
                if hasattr(module, 'weight'):
                    weight = module.weight()
                    new_scale = weight.q_scale() * 2  # 스케일을 조정하여 값의 범위를 4비트에 맞춤
                    new_zero_point = max(min(weight.q_zero_point(), 15), -16)
                    # 새로운 스케일과 제로 포인트로 가중치를 다시 양자화
                    module.weight().data = torch.quantize_per_tensor(weight.dequantize(), new_scale, new_zero_point, torch.qint8)

    elif quantization_type == 'qat':
        # Quantization-Aware Training (QAT)
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        torch.quantization.prepare_qat(model, inplace=True)
        # QAT를 위한 추가 훈련 단계가 필요할 수 있습니다.
        # 여기에서 QAT 훈련 과정을 구현할 수 있습니다.
        if num_bits == 8:
            # QAT 후 8-bit 양자화를 적용
            model_quantized = torch.quantization.convert(model.eval(), inplace=False)
        elif num_bits == 4:
            # QAT 후 모델을 8비트로 양자화하고, 4비트처럼 조정
            model_quantized = torch.quantization.convert(model.eval(), inplace=False)
            for name, module in model_quantized.named_modules():
                if hasattr(module, 'weight'):
                    weight = module.weight()
                    new_scale = weight.q_scale() * 2  # 스케일 조정
                    new_zero_point = max(min(weight.q_zero_point(), 15), -16)
                    module.weight().data = torch.quantize_per_tensor(weight.dequantize(), new_scale, new_zero_point, torch.qint8)

    model_quantized.to(device)
    return model_quantized

if __name__ == '__main__':
    main()
