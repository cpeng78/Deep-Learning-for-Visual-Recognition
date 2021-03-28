import torch
import torch.nn as nn

class StyleLoss(nn.Module):
    def gram_matrix(self, features, normalize=True):
        """
            Compute the Gram matrix from features.

            Inputs:
            - features: PyTorch Variable of shape (N, C, H, W) giving features for
              a batch of N images.
            - normalize: optional, whether to normalize the Gram matrix
                If True, divide the Gram matrix by the number of neurons (H * W * C)

            Returns:
            - gram: PyTorch Variable of shape (N, C, C) giving the
              (optionally normalized) Gram matrices for the N input images.
            """
        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # HINT: you may find torch.bmm() function is handy when it comes to process  #
        # matrix product in a batch. Please check the document about how to use it.  #
        ##############################################################################

        # (1) Reshape feature vector into shape N, C, H*W
        N, C, H, W = features.shape
        feature_vec = features.view(N, C, -1)  # out N, C, H*W

        # (2) Since we need to take dot product of each image in batch,
        # we need a modified transpoe of shape N, H*W, C
        # so permute axes to produce that vector
        feature_vec_permuted = feature_vec.permute(0, 2, 1)

        # (3) Batchwise multiplication
        # torch.bmm takes two tensors of (b, n, m) and (b, m, p)
        # and produces output of shape (b, n, p)
        gram = torch.bmm(feature_vec, feature_vec_permuted)  # output of shape N, C, C

        # (4) Normalise if flag is set to true
        if normalize:
            gram /= (H * W * C)

        return gram
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################
    def forward(self, feats, style_layers, style_targets, style_weights):
        """
           Computes the style loss at a set of layers.

           Inputs:
           - feats: list of the features at every layer of the current image, as produced by
             the extract_features function.
           - style_layers: List of layer indices into feats giving the layers to include in the
             style loss.
           - style_targets: List of the same length as style_layers, where style_targets[i] is
             a PyTorch Variable giving the Gram matrix the source style image computed at
             layer style_layers[i].
           - style_weights: List of the same length as style_layers, where style_weights[i]
             is a scalar giving the weight for the style loss at layer style_layers[i].

           Returns:
           - style_loss: A PyTorch Variable holding a scalar giving the style loss.
           """

        ##############################################################################
        # TODO: Implement content loss function                                      #
        # Please pay attention to use torch tensor math function to finish it.       #
        # Otherwise, you may run into the issues later that dynamic graph is broken  #
        # and gradient can not be derived.                                           #
        #                                                                            #
        # Hint:                                                                      #
        # you can do this with one for loop over the style layers, and should not be #
        # very much code (~5 lines). Please refer to the 'style_loss_test' for the   #
        # actual data structure.                                                     #
        #                                                                            #
        # You will need to use your gram_matrix function.                            #
        ##############################################################################

        # (0) Initialize style_loss tensor - it is actaully a scalar
        style_loss = torch.zeros(1)

        # (1) Extract only the features as specified by selected features
        # This can be done by list comprehension
        selected_feats = [feats[index] for index in style_layers]

        # (2) For each of these selected layers, do:
        for i, feature in enumerate(selected_feats):
            # Compute gram matrix of features of that layer
            style_original = self.gram_matrix(feature, normalize=True)

            # Compute L2 norm between computed gram and target gram
            l2_norm = (style_original - style_targets[i]).pow(2).sum()

            # Loss of that layer is the L2 norm times weights for layer
            layer_style_loss = style_weights[i] * l2_norm

            # Add loss of each layer to give total style loss
            style_loss += layer_style_loss

        return style_loss

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

