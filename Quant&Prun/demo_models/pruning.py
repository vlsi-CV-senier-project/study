import torch
import torchvision
import time

import torchvision.transforms as transforms
from torch import nn, optim

import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import torch.nn.utils.prune as prune

import argparse
from tqdm import tqdm

# 가지치기 적용시 연산량이 줄어들었는지 확인하기 위한 라이브러리
from thop import profile
import thop

class QAT_model(nn.Module):
    def __init__(self,model_fc):
        super(QAT_model,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model = model_fc
    def forward(self,x):
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x

# model 인스턴스화
# 사전학습된 ResNet50 모델 로드
pretrained_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# 모델의 마지막 fully connected layer를 CIFAR-100 데이터셋에 맞게 수정
pretrained_model.fc = torch.nn.Linear(pretrained_model.fc.in_features, 100)

# QAT_model 인스턴스화 및 모델 준비
model = QAT_model(pretrained_model)


def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with Pruning and Quantization')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--quant',default = 'dynamic',type = str, help = 'Quantization method',choices = ['dynamic', 'QAT','None'])
    parser.add_argument('--bits',default=8,type=int, choices = ['4', '8'])
    # select pruning 
    parser.add_argument('--prun',default = 'structured',type = str, choices = ['structured', 'magni','None'])
    parser.add_argument('--percent',default = 0.1,type = float, choices= [0.1,0.3,0.5])
    # channel wise pruning 여부 선택
    parser.add_argument('--channel',default = False, type = bool, choices=[True,False])
    args = parser.parse_args()
    return args

# magnitude based pruning
def magni_prune(model, percent, channel=False):
    for module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not channel:
                # Magnitude-based weight pruning
                # module의 weight에서 절대값에 대한 특정 분위수를 계산한다. 
                threshold = torch.quantile(torch.abs(module.weight.data), percent)
                # weight의 절대값이 임계값보다 크다면 boolean mask를 생성한다.  
                mask = torch.abs(module.weight.data) > threshold
                # 가중치 텐서에 마스크를 적용한다.
                
                module.weight.data *= mask
            # else:
            #     # Channel-wise pruning
            #     # weight 에 대하여 0번째 차원을 제외한 모든 차원에 대한 노름을 계산해 각 채널의 중요도를 측정
            #     # 각열의 2-norm 계산
            #     norms = torch.norm(module.weight.data, p=2, dim=[1, 2, 3])
            #     # norm = 각 채널의 중요도  threshold 보다 큰  norm을 제외하고 가지치기 진행
            #     threshold = torch.quantile(norms, percent)
            #     mask = norms > threshold
            #     module.weight.data *= mask[:, None, None, None]



# structured pruning
def struct_prune(model, percent, channel = False):
    for module in model. named_modules():
        # Conv2d 층에만 가지치기를 적용한다.
        if isinstance(module, nn.Conv2d):
            if not channel:
                # structured weight pruning
                # dim= 필터의 차원 n= n-norm dim = 0  채널단위, dim = 1 필터단위 
                prune.ln_structured(module, name='weight', amount=percent, n=1, dim=1)

            else:
                prune.ln_structured(module, name='weight', amount=percent, n=1, dim=0)
           
                



def main():
    args = parse_arguments()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    #Quantization bits
    if args.bits == 8:
        inttype = torch.qint8
    elif args.bits == 4:
        inttype = torch.qint4

    #Pruning percent

    # CIFAR-100 데이터셋 불러오기와 사전학습된 ResNet50 모델 로드 로직 여기에 포함
    # CIFAR-100 데이터셋 불러오기
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)

    # 사전학습된 ResNet50 모델 로드
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    model.to(device)

    # 모델의 마지막 fully connected layer를 CIFAR-100 데이터셋에 맞게 수정
    model.fc = torch.nn.Linear(model.fc.in_features, 100)
    model.fc.to(device)
    

    # 모델 가지치기 후 연산량 측정
    input = torch.randn(1, 3, 224, 224).to(device)  # 대상 모델에 맞는 입력 크기 사용 할 것!
    flops, params = profile(model, inputs=(input, ), verbose=False)
    print(f"Pruned model FLOPs: {flops}")
    print(f"Pruned model parameters: {params}")

    #Pruning
    if args.prun == 'magni':
        magni_prune(model, args.percent, channel=args.channel)
    elif args.prun == 'structured':
        struct_prune(model, args.percent, channel=args.channel)

        
    # 가지치기 후 연산량 측정을 다시 수행
    flops_after, params_after = profile(model, inputs=(input, ), verbose=False)
    print(f"Pruned model FLOPs after pruning: {flops_after}")
    print(f"Pruned model parameters after pruning: {params_after}")    

    #QAT 적용
    if(args.quant == 'QAT'):
        model = QAT_model(model)
        model.to(device)
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('x86')
        #model = torch.quantization.fuse_modules(model,[['conv', 'bn', 'relu','resnet50','fc']])
        # 모델 구조 병합용 코드, 없어도됨
        model = torch.quantization.prepare_qat(model.train())

    

    # 파인튜닝을 위한 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 감소 설정

    # 학습 로직
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        loop = tqdm(trainloader, leave=True)
        for i, data in enumerate(loop, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loop.set_description(f'Epoch [{epoch+1}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()
            if i % 100 == 99:  # print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                break
        scheduler.step()  # 학습률 감소 적용



    #Dynamic Quantization
    if args.quant == 'dynamic':
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear},
            dtype = inttype
        )
#        model.to(device)
    #QAT
    elif args.quant =='QAT':
        model.eval()
        model = torch.quantization.convert(model)
        model.to(device)


    # 모델을 평가 모드로 전환
    model.eval()
    # 정확도 평가
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')
    end_time = time.time()
    total_time = end_time - start_time
    average_inference_time = total_time / len(testset)
    print(f'Average inference time per image: {average_inference_time:.6f} seconds')

    # 모델 크기 계산    
    model_size = sum(param.numel() for param in model.parameters())
    print(f'Model size: {model_size} parameters')

if __name__ == '__main__':
    main()



