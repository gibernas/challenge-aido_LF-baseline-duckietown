import numpy as np

import torch
import torchvision

import networks as pix2pix_networks
from model import ConvSkip
from PIL import Image
import cv2

class TrimWrapper:
    def __init__(self):
        self.trim_est = 0

        self.k = 27
        self.b = 0.108
        self.R = 0.0318
        self.dt = 1/15

        self.model = self.load_model()

    def undistort_action(self, u_l, u_r):
        u_l = u_l*(1 - self.trim_est)
        u_r = u_r*(1 + self.trim_est)
        act = [u_l, u_r]
        return u_l, u_r

    def estimate_trim(self, x):
        # x = [delta_phi, u_l, u_r]
        # Remove data when DB accelerates
        x = x[30:]

        # Remove data when there is no delta angle
        mask = x[:, 0] != 0
        x = x[mask]

        # Consider ideal kinematics
        omega_l = x[:, 1] * self.k
        omega_r = x[:, 2] * self.k
        omega = (omega_r - omega_l) * self.R / self.b
        v = self.R / 2 * (omega_l + omega_r)

        # Compute an estimate for the real omega
        omega_meas = x[:, 0] / self.dt

        # Computing the trim paramter
        t = np.divide(omega_meas - omega, v) * self.b / 2
        t = np.expand_dims(t, 1)

        # Discard outliers
        mask = np.abs(t) < 1
        t = t[mask]

        # Compute average over samples
        t_est = np.mean(t)

        return t_est

    def get_delta_phi(self, img0,  img1):
        img0 = torch.from_numpy(img0)[None, None, :, :]
        img1 = torch.from_numpy(img1)[None, None, :, :]

        img = torch.cat((img0, img1), 1)

        delta_phi = self.model(img.double())
        return  delta_phi

    def load_model(self):
        checkpoint = torch.load('/workspace/ConvSkip.pth', map_location=torch.device('cpu'))
        model = checkpoint['model']
        model.load_state_dict(checkpoint['state_dict'])
        return model.eval()


class ImageTransformer:

    def __init__(self):

        self.load_model()

    def load_model(self):

        self.transform_net = pix2pix_networks.UnetGenerator(3, 3, 8).cpu()

        state_dict = torch.load("/workspace/latest_net_G.pth",
                                map_location="cpu")

        self.transform_net.load_state_dict(state_dict)

        self.transform_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            torchvision.transforms.Lambda(lambda x: x.unsqueeze(0)),
        ])

    def transform_img(self, img):
        img = self.transform_transform(img)
        img = self.transform_net(img)
        img = (img + 1) / 2.0 * 255.0
        img = img.clamp(0, 255).cpu().squeeze(0).numpy()
        img = img.transpose(1, 2, 0).astype("uint8")
        img = cv2.resize(img, (640,480))
        return img


if __name__ == '__main__':
    a = ImageTransformer()

    from PIL import Image
    import matplotlib.pyplot as plt
    img = np.ones((244,234,3))
    img = Image.open('/home/gianmarco/lenna1.png')
    img.show()
    with torch.no_grad():

        img = a.transform_img(img)

        plt.imshow(img)
        plt.show()