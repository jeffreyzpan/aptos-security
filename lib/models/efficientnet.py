from efficientnet_pytorch import EfficientNet

'''
Uses https://github.com/lukemelas/EfficientNet-PyTorch as a backend
'''

def efficientnetb0(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b0', num_classes=num_classes)
    return model

def efficientnetb1(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b1', num_classes=num_classes)
    return model

def efficientnetb2(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b2', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b2', num_classes=num_classes)
    return model

def efficientnetb3(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b3', num_classes=num_classes)
    return model

def efficientnetb4(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b4', num_classes=num_classes)
    return model

def efficientnetb5(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b5', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b5', num_classes=num_classes)
    return model

def efficientnetb6(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b6', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b6', num_classes=num_classes)
    return model

def efficientnetb7(num_classes=1000, pretrained=True, adv_train=False):
    if pretrained:
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes, advprop=adv_train)
    else:
        model = EfficientNet.from_name('efficientnet-b7', num_classes=num_classes)
    return model
