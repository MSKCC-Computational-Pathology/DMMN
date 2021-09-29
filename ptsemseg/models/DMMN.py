import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace

class DMMN(nn.Module):
    def __init__(self,n_classes):
        super(DMMN, self).__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / 2) for x in filters]

        # 20x
        self.conv1dn_20x = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool1_20x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv2dn_20x = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool2_20x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv3dn_20x = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool3_20x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv4dn_20x = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool4_20x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.center_20x = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans4_20x = nn.Sequential(
            nn.ConvTranspose2d(filters[4]+filters[3]+filters[2], filters[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv4up_20x = nn.Sequential(
            nn.Conv2d(filters[3]+filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans3_20x = nn.Sequential(
            nn.ConvTranspose2d(filters[3]+filters[2]+filters[1], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv3up_20x = nn.Sequential(
            nn.Conv2d(filters[2]+filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans2_20x = nn.Sequential(
            nn.ConvTranspose2d(filters[2]+filters[1]+filters[0], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv2up_20x = nn.Sequential(
            nn.Conv2d(filters[1]+filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans1_20x = nn.Sequential(
            nn.ConvTranspose2d(filters[1]+filters[0], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv1up_20x = nn.Sequential(
            nn.Conv2d(filters[0]+filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        # 10x
        self.conv1dn_10x = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool1_10x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv2dn_10x = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool2_10x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv3dn_10x = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool3_10x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv4dn_10x = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool4_10x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.center_10x = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans4_10x = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv4up_10x = nn.Sequential(
            nn.Conv2d(filters[3]+filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans3_10x = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv3up_10x = nn.Sequential(
            nn.Conv2d(filters[2]+filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans2_10x = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv2up_10x = nn.Sequential(
            nn.Conv2d(filters[1]+filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans1_10x = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv1up_10x = nn.Sequential(
            nn.Conv2d(filters[0]+filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        # 5x
        self.conv1dn_5x = nn.Sequential(
            nn.Conv2d(3, filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool1_5x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv2dn_5x = nn.Sequential(
            nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool2_5x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv3dn_5x = nn.Sequential(
            nn.Conv2d(filters[1], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool3_5x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.conv4dn_5x = nn.Sequential(
            nn.Conv2d(filters[2], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )
        self.maxpool4_5x = nn.MaxPool2d(kernel_size=2, return_indices=False)

        self.center_5x = nn.Sequential(
            nn.Conv2d(filters[3], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[4], filters[4], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans4_5x = nn.Sequential(
            nn.ConvTranspose2d(filters[4], filters[3], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv4up_5x = nn.Sequential(
            nn.Conv2d(filters[3]+filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[3], filters[3], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans3_5x = nn.Sequential(
            nn.ConvTranspose2d(filters[3], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv3up_5x = nn.Sequential(
            nn.Conv2d(filters[2]+filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[2], filters[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans2_5x = nn.Sequential(
            nn.ConvTranspose2d(filters[2], filters[1], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv2up_5x = nn.Sequential(
            nn.Conv2d(filters[1]+filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.convtrans1_5x = nn.Sequential(
            nn.ConvTranspose2d(filters[1], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            )
        self.conv1up_5x = nn.Sequential(
            nn.Conv2d(filters[0]+filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            )

        self.final = nn.Sequential(
            nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(filters[0], n_classes, kernel_size=1)
            )

    def forward(self, inputs_20x, inputs_10x, inputs_5x):
        conv1_5x = self.conv1dn_5x(inputs_5x)
        maxpool1_5x = self.maxpool1_5x(conv1_5x)

        conv2_5x = self.conv2dn_5x(maxpool1_5x)
        maxpool2_5x = self.maxpool2_5x(conv2_5x)

        conv3_5x = self.conv3dn_5x(maxpool2_5x)
        maxpool3_5x = self.maxpool3_5x(conv3_5x)

        conv4_5x = self.conv4dn_5x(maxpool3_5x)
        maxpool4_5x = self.maxpool4_5x(conv4_5x)

        center_5x = self.center_5x(maxpool4_5x)

        up4_5x = self.convtrans4_5x(center_5x)
        up4_5x = self.conv4up_5x(torch.cat([conv4_5x, up4_5x], 1))

        up3_5x = self.convtrans3_5x(up4_5x)
        up3_5x = self.conv3up_5x(torch.cat([conv3_5x, up3_5x], 1))

        up2_5x = self.convtrans2_5x(up3_5x)
        up2_5x = self.conv2up_5x(torch.cat([conv2_5x, up2_5x], 1))

        up1_5x = self.convtrans1_5x(up2_5x)
        up1_5x = self.conv1up_5x(torch.cat([conv1_5x, up1_5x], 1))

        conv1_10x = self.conv1dn_10x(inputs_10x)
        maxpool1_10x = self.maxpool1_10x(conv1_10x)

        conv2_10x = self.conv2dn_10x(maxpool1_10x)
        maxpool2_10x = self.maxpool2_10x(conv2_10x)

        conv3_10x = self.conv3dn_10x(maxpool2_10x)
        maxpool3_10x = self.maxpool3_10x(conv3_10x)

        conv4_10x = self.conv4dn_10x(maxpool3_10x)
        maxpool4_10x = self.maxpool4_10x(conv4_10x)

        center_10x = self.center_10x(maxpool4_10x)

        up4_10x = self.convtrans4_10x(center_10x)
        up4_10x = self.conv4up_10x(torch.cat([conv4_10x, up4_10x], 1))

        up3_10x = self.convtrans3_10x(up4_10x)
        up3_10x = self.conv3up_10x(torch.cat([conv3_10x, up3_10x], 1))

        up2_10x = self.convtrans2_10x(up3_10x)
        up2_10x = self.conv2up_10x(torch.cat([conv2_10x, up2_10x], 1))

        up1_10x = self.convtrans1_10x(up2_10x)
        up1_10x = self.conv1up_10x(torch.cat([conv1_10x, up1_10x], 1))

        conv1_20x = self.conv1dn_20x(inputs_20x)
        maxpool1_20x = self.maxpool1_20x(conv1_20x)

        conv2_20x = self.conv2dn_20x(maxpool1_20x)
        maxpool2_20x = self.maxpool2_20x(conv2_20x)

        conv3_20x = self.conv3dn_20x(maxpool2_20x)
        maxpool3_20x = self.maxpool3_20x(conv3_20x)

        conv4_20x = self.conv4dn_20x(maxpool3_20x)
        maxpool4_20x = self.maxpool4_20x(conv4_20x)

        center_20x = self.center_20x(maxpool4_20x)

        up3_5x_cropped = up3_5x[:,:,24:40,24:40]
        up4_10x_cropped = up4_10x[:,:,8:24,8:24]
        up4_20x = self.convtrans4_20x(torch.cat([center_20x,up4_10x_cropped,up3_5x_cropped],1))
        up4_20x = self.conv4up_20x(torch.cat([conv4_20x, up4_20x], 1))

        up2_5x_cropped = up2_5x[:,:,48:80,48:80]
        up3_10x_cropped = up3_10x[:,:,16:48,16:48]
        up3_20x = self.convtrans3_20x(torch.cat([up4_20x,up3_10x_cropped,up2_5x_cropped],1))
        up3_20x = self.conv3up_20x(torch.cat([conv3_20x, up3_20x], 1))

        up1_5x_cropped = up1_5x[:,:,96:160,96:160]
        up2_10x_cropped = up2_10x[:,:,32:96,32:96]
        up2_20x = self.convtrans2_20x(torch.cat([up3_20x,up2_10x_cropped,up1_5x_cropped],1))
        up2_20x = self.conv2up_20x(torch.cat([conv2_20x, up2_20x], 1))

        up1_10x_cropped = up1_10x[:,:,64:192,64:192]
        up1_20x = self.convtrans1_20x(torch.cat([up2_20x,up1_10x_cropped],1))
        up1_20x = self.conv1up_20x(torch.cat([conv1_20x, up1_20x], 1))

        final = self.final(up1_20x)
        final = F.log_softmax(final,dim=1)

        return final
