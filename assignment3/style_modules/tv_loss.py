import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def forward(self, img, tv_weight):
        """
            Compute total variation loss.

            Inputs:
            - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
            - tv_weight: Scalar giving the weight w_t to use for the TV loss.

            Returns:
            - loss: PyTorch Variable holding a scalar giving the total variation loss
              for img weighted by tv_weight.
            """

        ##############################################################################
        # TODO: Implement total variation loss function                              #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        ##############################################################################

        # Get all shapes
        _, C, H, W = img.shape

        ######### (i) Implementing column variation ##########

        # (1) Create two shifted versions of img with
        # left shifted image having last column as zero
        # right shifted image having first column as zero
        empty_column = torch.zeros(1, C, H, 1)

        img_shifted_left = torch.cat((img, empty_column), 3)
        img_shifted_right = torch.cat((empty_column, img), 3)

        # (2) Find the squared distance of these two shifted imgs
        l2_norm_col = (img_shifted_left - img_shifted_right).pow(2)

        # (3) Discard first and last columns to give output shape 1, C, H, W-1
        col_variation = l2_norm_col[:, :, :, 1:-1]

        # (4) Sum across channels, rows and columns
        col_variation_sum = col_variation.sum()

        ######### (ii) Implementing row variation #############

        # (1) Create two shifted versions of img with
        # top shifted image having last row as zero
        # bottom shifted image having first row as zero
        empty_row = torch.zeros(1, C, 1, W)

        img_shifted_top = torch.cat((img, empty_row), 2)
        img_shifted_bottom = torch.cat((empty_row, img), 2)

        # (2) Find the squared distance of these two shifted imgs
        l2_norm_row = (img_shifted_top - img_shifted_bottom).pow(2)

        # (3) Discard first and last rows to give output shape 1, C, H-1, W
        row_variation = l2_norm_row[:, :, 1:-1, :]

        # (4) Sum across channels, rows and columns
        row_variation_sum = row_variation.sum()

        ######### (iii) Computing TV reg term #############

        loss = tv_weight * (row_variation_sum + col_variation_sum)

        return loss

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################