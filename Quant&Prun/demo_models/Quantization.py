import torch
import torchvision
import time
import torchvision.transforms as transforms
from torch import nn, optim
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
import argparse
from tqdm import tqdm

from torchvision.models.resnet import Bottleneck, ResNet

class CustomBottleneck(Bottleneck):
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(CustomBottleneck, self).__init__(inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
        # 여기에 커스텀 코드 추가
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        # forward 함수를 커스터마이징 할 수 있습니다.
        # 기본적인 Bottleneck의 forward 경로를 따르거나 새로운 로직을 추가할 수 있습니다.
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        #out += identity
        out = self.skip_add.add(identity, out)
        out = self.relu(out)

        return out

class CustomResNet50(ResNet):
            def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                # CustomResNet50에서는 CustomBottleneck을 block으로 사용합니다.
                return super()._make_layer(CustomBottleneck, planes, blocks, stride, dilate)

class QAT_model(nn.Module):
    def __init__(self,model_fc):
        super(QAT_model,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.resnet = model_fc
    def forward(self,x):
        x = self.quant(x)
        x = self.resnet(x)
        x = self.dequant(x)
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with Pruning and Quantization')
    parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--quant',default = 'dynamic',type = str, help = 'Quantization method',choices = ['dynamic', 'QAT','None'])
    parser.add_argument('--bits',default=8,type=int, choices = [4, 8])
    parser.add_argument('--prun',default = 'struct',type = str, choices = ['struct', 'magni','None'])
    parser.add_argument('--percent',default = 10,type = int, choices= [10,30,50])

    args = parser.parse_args()
    return args

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

    #QAT 적용
    if(args.quant == 'QAT'):

        # 2. 사전 학습된 ResNet50 모델 로드
        resnet50_pretrained = models.resnet50(pretrained=True)
        # 3. 모델의 Bottleneck 교체
        # ResNet 모델의 _make_layer 메소드에서 Bottleneck 클래스를 사용하는데, 이를 CustomBottleneck으로 교체합니다.
        # 사전 학습된 모델의 가중치를 새로운 모델 구조에 복사합니다. (필요한 경우)
        model = CustomResNet50(models.resnet.Bottleneck, [3, 4, 6, 3])
        model.load_state_dict(resnet50_pretrained.state_dict(), strict=False)
        model.to(device)

        model = QAT_model(model)
        model.eval()
        qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        model.qconfig = qconfig
        model = torch.ao.quantization.prepare_qat(model.train(),inplace = True)                                                        
    else:   
        model.train()
    #Pruning
    

    # 파인튜닝을 위한 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)  # 학습률 감소 설정

    # 학습 로직
    #model.train()    위에 선언
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
            if i == 10:  # print every 100 mini-batches
                break
        print('[epoch : %d] loss: %.3f' %(epoch + 1, running_loss / 100))
        scheduler.step()  # 학습률 감소 적용


    # 모델을 평가 모드로 전환
    
    #pruning
    if args.prun == 'staic':
        #apply pruning
        #train again
        print("sample")

    #Dynamic Quantization
    if args.quant == 'dynamic':
        device = "cpu"
        model.to(device)
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype = inttype
        )
    #QAT
    elif args.quant =='QAT':
        device = "cpu"
        model.to(device)
        model.eval()
        model = torch.ao.quantization.convert(model,inplace = True)
        model.to(device)
        
        

    # 정확도 평가
    correct = 0
    total = 0
    length = len(testloader) #79개
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if((args.quant != "None") & (i == 10)):
                length = i
                break
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy}%')
    end_time = time.time()
    total_time = end_time - start_time
    average_inference_time = total_time / length
    print(f'Average inference time per image: {average_inference_time:.6f} seconds')

    # 모델 크기 계산    
    model_size = sum(param.numel() for param in model.parameters())
    print(f'Model size: {model_size} parameters')

if __name__ == '__main__':
    main()
