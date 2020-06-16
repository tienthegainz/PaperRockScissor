from mobilenet_v2 import mobilenet_v2, MobileNetV2
from PIL import Image
from torchvision import transforms
import torch
import torch.nn.functional as F


def preprocess_image(pil_image):
    val_tfms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = val_tfms(pil_image)
    image = image.unsqueeze_(0).cpu()
    # image /= 255.00
    image = F.interpolate(image, size=256)
    return image


if __name__ == "__main__":
    ########################### Magic code line #################################
    convert = {0: 'la', 1: 'dam', 2: 'keo'}
    checkpoint = torch.load('MBN_epoch_1_loss_0.10.pth',
                            map_location=torch.device('cpu'))
    # print(checkpoint)
    model = MobileNetV2(num_classes=3)
    model.load_state_dict(checkpoint)
    # print(model)
    model.eval()
    # Your image here
    pil_image1 = Image.open('test_data/dam.jpg')
    image1 = preprocess_image(pil_image1)
    # print(image.shape)
    output = model(image1)
    # print(output)
    _, predicted = torch.max(output.data, 1)
    print(convert[int(predicted)])
    # Your image here
    pil_image1 = Image.open('test_data/keo.jpg')
    image1 = preprocess_image(pil_image1)
    # print(image.shape)
    output = model(image1)
    # print(output)
    _, predicted = torch.max(output.data, 1)
    print(convert[int(predicted)])
    # Your image here
    pil_image1 = Image.open('test_data/la.jpg')
    image1 = preprocess_image(pil_image1)
    # print(image.shape)
    output = model(image1)
    # print(output)
    _, predicted = torch.max(output.data, 1)
    print(convert[int(predicted)])
