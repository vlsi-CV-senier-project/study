import torch
import torchvision
import time
import torchvision.transforms as transforms
from torch import nn, optim, Tensor
import torchvision.models as models
import argparse
from tqdm import tqdm

import torch.nn.utils.prune as prune
import torch.ao.quantization as quantization
import torchvision.models.quantization as quantized_models



#new class
import warnings

class QAT_model(nn.Module):
    def __init__(self):
        super(QAT_model,self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.mobilenet = quantized_models.QuantizableMobileNetV2()
        self.fc = nn.Linear(1000,100)
    def forward(self,x):
        x = self.quant(x)
        x = self.mobilenet(x)
        x = self.fc(x)
        x = self.dequant(x)
        return x
    
class out_model(nn.Module):
    def __init__(self,model):
        super(out_model,self).__init__()
        self.mobilenet = model
        self.fc = nn.Linear(1000,100)
    def forward(self,x):
        x = self.mobilenet(x)
        x = self.fc(x)
        return x

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR-100 Training with Pruning and Quantization')
    parser.add_argument('--epochs', default=30, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=128, type=int, help='mini-batch size (default: 128)')
    parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--quant',default = 'QAT',type = str, help = 'Quantization method',choices = ['dynamic', 'QAT','None'])
    parser.add_argument('--bits',default=8,type=int, choices = [4, 8])

    parser.add_argument('--prun',default = 'structured',type = str, choices = ['structured', 'magni','None'])
    parser.add_argument('--percent',default = 0.5,type = float, choices= [0.1,0.3,0.5])
    # channel wise pruning 여부 선택
    parser.add_argument('--channel',default = True, type = bool, choices=[True,False])
    parser.add_argument('--test',default = False, type = bool, choices=[True,False])

    args = parser.parse_args()
    return args

# structured pruning
def struct_prune(model, percent, channel = False):
    for module in model. named_modules():
        if isinstance(module, nn.Conv2d):
            if not channel:
                # structured weight pruning
                prune.ln_structured(module, name='weight', amount=percent, n=1, dim=1)

            else:
                prune.ln_structured(module, name='weight', amount=percent, n=1, dim=0)
           
def magni_prune(model, percent, channel=False):
    for module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            if not channel:
                # Magnitude-based weight pruning
                threshold = torch.quantile(torch.abs(module.weight.data), percent)
                mask = torch.abs(module.weight.data) > threshold
                module.weight.data *= mask
            else:
                # Channel-wise pruning
                norms = torch.norm(module.weight.data, p=2, dim=[1, 2, 3])
                threshold = torch.quantile(norms, percent)
                mask = norms > threshold
                module.weight.data *= mask[:, None, None, None]



def main():
    args = parse_arguments()
    print(f"Pruning Method : {args.prun} with {args.percent}%, channel wise Pruning : {args.channel}\nQuantization Method : {args.quant}\n\n")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


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

    if args.quant == 'QAT':
        model = QAT_model()
        model.to(device)
    else :
        model = models.mobilenet_v2(pretrained = True)
        model = out_model(model)
        model.to(device)
    
    if args.prun == 'magni':
        magni_prune(model, args.percent, channel=args.channel)
    elif args.prun == 'structured':
        struct_prune(model, args.percent, channel=args.channel)

    
    #QAT 적용
    if(args.quant == 'QAT'):    #QAT에 사용되는 모델은 아예 다름
        #model.fuse_model()
        model.eval()
        qconfig = quantization.get_default_qat_qconfig('x86') #fbgemm', 'x86', 'qnnpack', 'onednn'
        model.qconfig = qconfig
        model = quantization.prepare_qat(model.train(),inplace = True)                                                        
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
            if (args.test == True and i == 5):
                break
            loop.set_description(f'Epoch [{epoch+1}/{args.epochs}]')
            loop.set_postfix(loss=loss.item())
            running_loss += loss.item()
        print("\n----Train Result---")
        print('[epoch : %d] loss: %.5f' %(epoch + 1, running_loss / 100))
        scheduler.step()  # 학습률 감소 적용

    # 모델을 평가 모드로 전환

    #QAT
    if args.quant =='QAT':
        device = "cpu"
        model.to(device)
        model.eval()
        model = quantization.convert(model,inplace = True)
        
    elif args.quant == 'dynamic':
        device = "cpu"
        model.to(device)
        model = torch.ao.quantization.quantize_dynamic(
            model,
            {nn.Linear},
            dtype = torch.qint8
        )

    # 정확도 평가
    correct = 0
    total = 0
    length = len(testloader) #79개
    start_time = time.time()
    with torch.no_grad():
        for i, data in enumerate(testloader, 0):
            images, labels = data[0].to(device,dtype=torch.float), data[1].to(device,dtype=torch.float)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            if (args.test == True and i == 5):
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
    print(f'Model size: {model_size:,} parameters')
    non_zero_param_count = sum(torch.count_nonzero(param).item() for param in model.parameters())
    print(f'Non_Zero_Parameters : {non_zero_param_count:,}')
    lt = time.localtime(time.time())
    save_time = time.strftime("%d%H%M",lt)
    torch.save(model.state_dict(), './models/'+'M_'+save_time+'.pt')



if __name__ == '__main__':
    main()
