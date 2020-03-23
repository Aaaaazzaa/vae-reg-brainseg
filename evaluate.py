import torch
from model import vaereg

from torch.utils.data import DataLoader
from datasets.data_loader import BraTSDataset

from utils import _validate
from losses import losses
device = torch.device('cuda')
model = vaereg.UNet()
checkpoint = torch.load('./checkpoints/downsampled4/downsampled4', map_location='cuda:0')
# checkpoint = torch.load('./checkpoints/baseline/baseline', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
model.eval()
model = model.to(device)

brats_data = BraTSDataset('./data/brats2018downsampled/test', dims=[32, 32, 32], modes=["t1", "t1ce", "t2", "flair"], downsample=4)
dataloader = DataLoader(brats_data, batch_size=1, shuffle=True, num_workers=0)

_validate(model, losses.DiceLoss(), dataloader, device, False)
