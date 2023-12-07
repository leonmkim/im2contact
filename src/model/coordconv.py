import torch
import torch.nn as nn
import torch.nn.modules.conv as conv
import torch.nn.functional as F

# From https://github.com/walsvid/CoordConv/blob/master/coordconv.py
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

class AddCoords(nn.Module):
    def __init__(self, rank, bb_h, bb_w, im_h, im_w, with_r=False, use_cuda=True):
        super(AddCoords, self).__init__()
        self.rank = rank
        # with r adds a third channel with the distance to the center of the image in l2 norm
        self.with_r = with_r
        self.use_cuda = use_cuda
        self.bb_h = bb_h
        self.bb_w = bb_w
        self.im_h = im_h
        self.im_w = im_w

    def forward(self, input_tensor, coord_pos):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        if self.rank == 1:
            batch_size_shape, channel_in_shape, dim_x = input_tensor.shape
            xx_range = torch.arange(dim_x, dtype=torch.int32)
            xx_channel = xx_range[None, None, :]

            xx_channel = xx_channel.float() / (dim_x - 1)
            xx_channel = xx_channel * 2 - 1
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
            out = torch.cat([input_tensor, xx_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 2:
            batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
            original_x = self.bb_w
            original_y = self.bb_h

            normalize_x = self.im_w
            normalize_y = self.im_h
            
            for ind in range(batch_size_shape):
                xx_ones = torch.ones([1, 1, 1, original_x], dtype=torch.int32)
                yy_ones = torch.ones([1, 1, 1, original_y], dtype=torch.int32)

                xx_range = torch.arange(original_y, dtype=torch.int32)
                yy_range = torch.arange(original_x, dtype=torch.int32)
                xx_range = xx_range[None, None, :, None]
                yy_range = yy_range[None, None, :, None]

                # https://discuss.pytorch.org/t/it-is-recommended-to-use-source-tensor-clone-detach-or-sourcetensor-clone-detach-requires-grad-true/101218/2
                xx_channel = torch.matmul(xx_range, xx_ones).to(device=device, dtype=torch.float) + coord_pos[ind][1].clone().detach().requires_grad_(True) #y_pos
                yy_channel = torch.matmul(yy_range, yy_ones).to(device=device, dtype=torch.float) + coord_pos[ind][0].clone().detach().requires_grad_(True) #x_pos

                # transpose y
                yy_channel = yy_channel.permute(0, 1, 3, 2)

                # Normalize to (-1, 1)
                xx_channel = xx_channel.float() / (normalize_y - 1)
                yy_channel = yy_channel.float() / (normalize_x - 1)

                xx_channel = xx_channel * 2 - 1
                yy_channel = yy_channel * 2 - 1

                if ind == 0:
                    xx_channel_all = xx_channel #16, 1, dim_y, dim_x
                    yy_channel_all = yy_channel
                    continue
                xx_channel_all = torch.cat([xx_channel_all, xx_channel], dim = 0)
                yy_channel_all = torch.cat([yy_channel_all, yy_channel], dim = 0)

            # xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
            # yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel_all = F.interpolate(xx_channel_all,[dim_y, dim_x] ).cuda()
                yy_channel_all = F.interpolate(yy_channel_all,[dim_y, dim_x] ).cuda()
            out = torch.cat([input_tensor, xx_channel_all, yy_channel_all], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel_all - 0.5, 2) + torch.pow(yy_channel_all - 0.5, 2))
                out = torch.cat([out, rr], dim=1)

        elif self.rank == 3:
            batch_size_shape, channel_in_shape, dim_z, dim_y, dim_x = input_tensor.shape
            xx_ones = torch.ones([1, 1, 1, 1, dim_x], dtype=torch.int32)
            yy_ones = torch.ones([1, 1, 1, 1, dim_y], dtype=torch.int32)
            zz_ones = torch.ones([1, 1, 1, 1, dim_z], dtype=torch.int32)

            xy_range = torch.arange(dim_y, dtype=torch.int32)
            xy_range = xy_range[None, None, None, :, None]

            yz_range = torch.arange(dim_z, dtype=torch.int32)
            yz_range = yz_range[None, None, None, :, None]

            zx_range = torch.arange(dim_x, dtype=torch.int32)
            zx_range = zx_range[None, None, None, :, None]

            xy_channel = torch.matmul(xy_range, xx_ones)
            xx_channel = torch.cat([xy_channel + i for i in range(dim_z)], dim=2)
            xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            yz_channel = torch.matmul(yz_range, yy_ones)
            yz_channel = yz_channel.permute(0, 1, 3, 4, 2)
            yy_channel = torch.cat([yz_channel + i for i in range(dim_x)], dim=4)
            yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            zx_channel = torch.matmul(zx_range, zz_ones)
            zx_channel = zx_channel.permute(0, 1, 4, 2, 3)
            zz_channel = torch.cat([zx_channel + i for i in range(dim_y)], dim=3)
            zz_channel = zz_channel.repeat(batch_size_shape, 1, 1, 1, 1)

            if torch.cuda.is_available and self.use_cuda:
                input_tensor = input_tensor.cuda()
                xx_channel = xx_channel.cuda()
                yy_channel = yy_channel.cuda()
                zz_channel = zz_channel.cuda()
            out = torch.cat([input_tensor, xx_channel, yy_channel, zz_channel], dim=1)

            if self.with_r:
                rr = torch.sqrt(torch.pow(xx_channel - 0.5, 2) +
                                torch.pow(yy_channel - 0.5, 2) +
                                torch.pow(zz_channel - 0.5, 2))
                out = torch.cat([out, rr], dim=1)
        else:
            raise NotImplementedError

        return out.float()


class CoordConv1d(conv.Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv1d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 1
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv1d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.AddCoords(input_tensor)
        out = self.conv(out)

        return out


class CoordConv2d(conv.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv2d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 2
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv2d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out

class CoordConv3d(conv.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, with_r=False, use_cuda=True):
        super(CoordConv3d, self).__init__(in_channels, out_channels, kernel_size,
                                          stride, padding, dilation, groups, bias)
        self.rank = 3
        self.addcoords = AddCoords(self.rank, with_r, use_cuda=use_cuda)
        self.conv = nn.Conv3d(in_channels + self.rank + int(with_r), out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        out = self.conv(out)

        return out