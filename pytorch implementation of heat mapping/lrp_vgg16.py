import cv2
import numpy
import utils
import torch
import torchvision

def lrp_heatmap(img):

    mean = torch.Tensor([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    std  = torch.Tensor([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    X = (torch.FloatTensor(img[numpy.newaxis].transpose([0,3,1,2])*1) - mean) / std
    model = torchvision.models.vgg16(pretrained=True); model.eval()
    layers = list(model._modules['features']) + utils.toconv(list(model._modules['classifier']))
    L = len(layers)

    A = [X]+[None]*L
    for l in range(L): A[l+1] = layers[l].forward(A[l])

    scores = numpy.array(A[-1].data.view(-1))
    ind = numpy.argsort(-scores)
    
    T = torch.FloatTensor((1.0*(numpy.arange(1000)==ind[0]).reshape([1,1000,1,1])))
    R = [0.0]*L + [(A[-1]*T).data]

    for l in range(1,L)[::-1]:    
        A[l] = (A[l].data).requires_grad_(True)
        if isinstance(layers[l],torch.nn.MaxPool2d): layers[l] = torch.nn.AvgPool2d(2)
        if isinstance(layers[l],torch.nn.Conv2d) or isinstance(layers[l],torch.nn.AvgPool2d):
            if l <= 16:       rho = lambda p: p + 0.25*p.clamp(min=0); incr = lambda z: z+1e-9
            if 17 <= l <= 30: rho = lambda p: p;                       incr = lambda z: z+1e-9+0.25*((z**2).mean()**.5).data
            if l >= 31:       rho = lambda p: p;                       incr = lambda z: z+1e-9
            z = incr(utils.newlayer(layers[l],rho).forward(A[l]))  # step 1
            s = (R[l+1]/z).data                                    # step 2
            (z*s).sum().backward(); c = A[l].grad                  # step 3
            R[l] = (A[l]*c).data                                   # step 4        
        else:
            R[l] = R[l+1]
    
    A[0] = (A[0].data).requires_grad_(True)

    lb = (A[0].data*0+(0-mean)/std).requires_grad_(True)
    hb = (A[0].data*0+(1-mean)/std).requires_grad_(True)

    z = layers[0].forward(A[0]) + 1e-9                                     # step 1 (a)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(min=0)).forward(lb)    # step 1 (b)
    z -= utils.newlayer(layers[0],lambda p: p.clamp(max=0)).forward(hb)    # step 1 (c)
    s = (R[1]/z).data                                                      # step 2
    (z*s).sum().backward(); c,cp,cm = A[0].grad,lb.grad,hb.grad            # step 3
    R[0] = (A[0]*c+lb*cp+hb*cm).data

    #return (utils.heatmap(numpy.array(R[0][0]).sum(axis=0),3.5,3.5))
    return numpy.array(R[l][0]).sum(axis=0)
