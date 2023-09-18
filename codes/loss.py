import torch
import torch.nn as nn


### For Numpy arrays:
def dice_loss(output, target, smooth=1e-7):
    intersection = (output*target).sum()
    union = (output+target).sum()

    loss = 1.0 - (2.0*intersection + smooth)/(union + smooth)
    return loss



### For Tensors:
class DiceLoss(nn.Module):
    def __init__(self, soft=False, smooth=1e-7, reduce=torch.mean):
        super().__init__()
        self.soft = soft
        self.smooth = smooth
        self.reduce = reduce
        self.__name__ = 'soft_dice' if soft else 'dice_loss'
        self.__name__ += '_batch' if not reduce else ''

    def forward(self, output, target):
        dims = (2,3)
        if not self.soft:
            output = torch.round(output)
        intersection = torch.sum(output*target, dims)
        union = torch.sum(output+target, dims)

        if self.reduce:
            loss = 1.0 - (2.0*intersection + self.smooth)/(union + self.smooth)
            return self.reduce(loss)
        else:
            intersection = torch.sum(intersection)
            union = torch.sum(union)
            return 1.0 - (2.0*intersection + self.smooth)/(union + self.smooth)



class IoULoss(nn.Module):
    def __init__(self, soft=False, smooth=1e-7, reduce=torch.mean):
        super().__init__()
        self.soft = soft
        self.smooth = smooth
        self.reduce = reduce
        self.__name__ = 'soft_IoU' if soft else 'IoU'
        self.__name__ += '_batch' if not reduce else ''

    def forward(self, output, target):
        dims = (2,3)
        output = torch.round(output)
        intersection = torch.sum(output*target, dims)
        union = torch.sum(output+target, dims) - intersection

        if self.reduce:
            loss = 1.0 - (intersection + self.smooth)/(union + self.smooth)
            return self.reduce(loss)
        else: 
            intersection = torch.sum(intersection)
            union = torch.sum(union)
            return 1.0 - (intersection + self.smooth)/(union + self.smooth)



class BCELoss(nn.Module):
    def __init__(self, weights=(1,1), reduce=torch.mean, clamp=True):
        super().__init__()
        self.weights = weights
        self.reduce = reduce
        self.clamp = clamp
        self.__name__ = 'bce'

    def forward(self, output, target):
        output = torch.clamp(output, min=1e-7, max=1-1e-7)
        log0, log1 = torch.log(output), torch.log(1-output)

        # nn.BCELoss clamps its log function outputs to be greater than or equal to -100
        # to always have a finite loss value and a linear backward method.
        if self.clamp:
            log0, log1 = torch.clamp(log0, min=-100.0), torch.clamp(log1, min=-100.0)

        loss = self.weights[0]*target*log0 + self.weights[1]*(1-target)*log1
        return self.reduce(-loss)




class FocalLoss(nn.Module):
    """
    -alpha_t(1-p_t)^gamma log(p_t)
    when gamma=1, Focal loss works like CE loss.
    """
    def __init__(self, alpha=(1,1), gamma=1.5, reduce=torch.mean, clamp=True):
        super().__init__()
        self.__name__ = 'focal'
        self.alpha = alpha
        self.gamma = gamma
        self.reduce = reduce
        self.clamp = True

    def forward(self, output, target):
        output = torch.clamp(output, min=1e-7, max=1-1e-7)
        log0, log1 = torch.log(output), torch.log(1-output)
        if self.clamp:
            log0, log1 = torch.clamp(log0, min=-100.0), torch.clamp(log1, min=-100.0)
        bce_loss = target*log0 + (1-target)*log1

        p_t = target*output + (1-target)*(1-output)
        alpha_t = target*self.alpha[1] + (1-target)*self.alpha[0]
        loss = alpha_t * (1-p_t)**self.gamma * bce_loss
        return self.reduce(-loss)



class torchBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = '(torch)bce'

    def forward(self, output, target):
        bce = torch.nn.BCELoss()
        return bce(output, target)



# Combo Losses
class BceDiceLoss(nn.Module):
    """
    alpha: alpha*BCE + (1-alpha)*Dice
    beta: the weight for BCE loss
    """
    def __init__(self, alpha=0.5, beta=(1,1), soft=False):
        super().__init__()
        self.__name__ = 'bce_soft_dice' if soft else 'bce_dice'
        self.alpha = alpha
        self.beta = beta
        self.soft = soft

    def forward(self, output, target):
        bce = BCELoss(weights=self.beta)
        dice = DiceLoss(self.soft)
        return self.alpha*bce(output, target) + (1-self.alpha)*dice(output, target)



class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, soft=False):
        super().__init__()
        self.__name__ = 'focal_soft_dice' if soft else 'focal_dice'
        self.alpha = alpha
        self.gamma = gamma
        self.soft = soft

    def forward(self, output, target):
        focal = FocalLoss(self.alpha, self.gamma)
        dice = DiceLoss(self.soft)
        return focal(output,target) + dice(output,target)

        
class torchComboLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.__name__ = 'dice_(torch)bce'

    def forward(self, output, target):
        bce = torch.nn.BCELoss()
        dice = DiceLoss(soft=False)
        return bce(output, target) + dice(output, target)



# Weighted Losses
class weightedDiceLoss(nn.Module):
    def __init__(self, weights=(1,1), smooth=1e-7):
        super().__init__()
        self.weights = weights
        self.smooth = smooth
        self.__name__ = 'weighted_dice'

    def forward(self, output, target):
        # Dice Loss
        pred = torch.round(output)
        intersection = torch.sum(pred*target, (2,3))
        union = torch.sum(pred+target, (2,3))

        loss = 1.0 - (2.0*intersection + self.smooth)/(union + self.smooth)

        # Applying weights
        weights = torch.tensor(self.weights).to(target.device)
        target_labels = torch.count_nonzero(target, dim=(1,2,3)).clamp(max=1)
        loss_weights = weights[target_labels]

        return torch.mean(loss_weights*loss)



class weightedFocalDiceLoss(nn.Module):
    """
    alpha: alpha*BCE + (1-alpha)*Dice
    beta: the weight for BCE loss
    """
    def __init__(self, alpha=0.5, gamma=1.5, weights=(1,1), smooth=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weights = weights
        self.smooth = smooth
        self.__name__ = 'weighted_focal_dice'

    def forward(self, output, target):
        # Focal Loss
        clamped = torch.clamp(output, min=1e-7, max=1-1e-7)
        log0, log1 = torch.log(clamped), torch.log(1-clamped)
        log0, log1 = torch.clamp(log0, min=-100.0), torch.clamp(log1, min=-100.0)
        bce_loss = self.weights[0]*target*log0 + self.weights[1]*(1-target)*log1

        p_t = target*output + (1-target)*(1-output)
        alpha_t = target*1 + (1-target)*self.alpha
        loss1 = -torch.mean(alpha_t * (1-p_t)**self.gamma * bce_loss, (2,3))

        # Dice Loss
        pred = torch.round(output)
        intersection = torch.sum(pred*target, (2,3))
        union = torch.sum(pred+target, (2,3))

        loss2 = 1.0 - (2.0*intersection + self.smooth)/(union + self.smooth)

        # Applying weights
        weights = torch.tensor(self.weights).to(target.device)
        target_labels = torch.count_nonzero(target, dim=(1,2,3)).clamp(max=1)
        loss_weights = weights[target_labels]

        return torch.mean(loss_weights*(loss1+loss2))



class weightedBceDiceLoss(nn.Module):
    """
    alpha: alpha*BCE + (1-alpha)*Dice
    beta: the weight for BCE loss
    """
    def __init__(self, alpha=0.5, weights=(1,1), smooth=1e-7):
        super().__init__()
        self.alpha = alpha
        self.weights = weights
        self.smooth = smooth
        self.__name__ = 'weighted_bce_dice'

    def forward(self, output, target):
        # BCE Loss
        clamped = torch.clamp(output, min=1e-7, max=1-1e-7)
        log0, log1 = torch.log(clamped), torch.log(1-clamped)
        log0, log1 = torch.clamp(log0, min=-100.0), torch.clamp(log1, min=-100.0)

        loss1 = -torch.mean(target*log0 + (1-target)*log1, (2,3))

        # Dice Loss
        pred = torch.round(output)
        intersection = torch.sum(pred*target, (2,3))
        union = torch.sum(pred+target, (2,3))

        loss2 = 1.0 - (2.0*intersection + self.smooth)/(union + self.smooth)

        # Applying weights
        weights = torch.tensor(self.weights).to(target.device)
        target_labels = torch.count_nonzero(target, dim=(1,2,3)).clamp(max=1)
        loss_weights = weights[target_labels]

        return torch.mean(loss_weights*(loss1+loss2))