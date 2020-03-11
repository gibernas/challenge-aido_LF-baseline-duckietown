import torch
from torchvision import transforms
from PIL import Image

from utils import TransCropHorizon
from models import VanillaCNN # , SpectralDropoutEasyCNN
device = torch.device("cpu")


class Model(object):
    def __init__(self):
        super(Model, self).__init__()

        image_res = 64

        self.transformsCustom = transforms.Compose([transforms.Resize(image_res),
                                                    TransCropHorizon(0.5, set_black=False),
                                                    transforms.Grayscale(num_output_channels=1),
                                                    transforms.ToTensor()])

        # model_name = 'SpectralDropoutCNN_1579294275.6305485_lr_0.001_bs_16_dataset_sim_totepo_200final.pt'
        model_name = 'VanillaCNN_1579294019.6894116_lr_0.001_bs_16_dataset_sim_totepo_200final.pt'
        model_path = '/workspace/' + model_name
        self.model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.double().to(device=torch.device('cpu'))

    def close(self):
        # TODO: release resources
        pass

    def predict(self, images):
        images = self.transformsCustom(Image.fromarray(images.astype('uint8')))
        images = images.double().to(device=torch.device('cpu'))
        images = images.unsqueeze(1)
        pose_estimates = self.model(images)

        return pose_estimates
