import torch
from torch.autograd import Function as TorchFunc
import numpy as np
from PIL import Image
import torch.nn.functional as F


# The ’deconvolution’ is equivalent to a backward pass through the network, except that 
# when propagating through a nonlinearity, its gradient is solely computed based on the 
# top gradient signal, ignoring the bottom input. In case of the ReLU nonlinearity this 
# amounts to setting to zero certain entries based on the top gradient. We propose to 
# combine these two methods: rather than masking out values corresponding to negative 
# entries of the top gradient (’deconvnet’) or bottom data (backpropagation), we mask 
# out the values for which at least one of these values is negative.

class CustomReLU(TorchFunc):
    """
    Define the custom change to the standard ReLU function necessary to perform guided backpropagation.
    We have already implemented the forward pass for you, as this is the same as a normal ReLU function.
    """

    @staticmethod
    def forward(self, x):
        output = torch.addcmul(torch.zeros(x.size()), x, (x > 0).type_as(x))
        self.save_for_backward(x, output)
        return output

    @staticmethod
    def backward(self, y):
        ##############################################################################
        # TODO: Implement this function. Perform a backwards pass as described in    #
        # the guided backprop paper ( there is also a brief description at the top   #
        # of this page).                                                             #
        # Note: torch.addcmul might be useful, and you can access  the input/output  #
        # from the forward pass with self.saved_tensors.                             #
        ##############################################################################
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(y)
        positive_mask_2 = (y > 0).type_as(y)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), y,
                                                 positive_mask_1), positive_mask_2)
        return grad_input
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################


class GradCam:
    def guided_backprop(self, X_tensor, y_tensor, gc_model):
        """
        Compute a guided backprop visualization using gc_model for images X_tensor and 
        labels y_tensor.

        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the guided backprop.

        Returns:
        - guided backprop: A numpy of shape (N, H, W, 3) giving the guided backprop for 
        the input images.
        """

        # Thanks to Farrukh Rahman (Fall 2020) for pointing out that Squeezenet has
        #  some of it's ReLU modules as submodules of 'Fire' modules
        #  
        for param in gc_model.parameters():
            param.requires_grad = True

        for idx, module in gc_model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                gc_model.features._modules[idx] = CustomReLU.apply
            elif module.__class__.__name__ == 'Fire':
                for idx_c, child in gc_model.features[int(idx)].named_children():
                    if child.__class__.__name__ == 'ReLU':
                        gc_model.features[int(idx)]._modules[idx_c] = CustomReLU.apply
        ##############################################################################
        # TODO: Implement guided backprop as described in paper.                     #
        # (Hint): Now that you have implemented the custom ReLU function, this       #
        # method will be similar to a single training iteration.                     #
        #                                                                            #
        # Also note that the output of this function is a numpy.                     #
        ##############################################################################
        output = gc_model(X_tensor)

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][y_tensor] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = X_tensor.grad.cpu().data.permute(0, 2, 3, 1).numpy()
        #output = output
        #print(output.shape)
        return output
        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

    def grad_cam(self, X_tensor, y_tensor, gc_model):
        """
        Input:
        - X_tensor: Input images; Tensor of shape (N, 3, H, W)
        - y: Labels for X; LongTensor of shape (N,)
        - model: A pretrained CNN that will be used to compute the gradcam.
        """
        conv_module = gc_model.features[12]
        self.gradient_value = None  # Stores gradient of the module you chose above during a backwards pass.
        self.activation_value = None  # Stores the activation of the module you chose above during a forwards pass.

        def gradient_hook(a, b, gradient):
            self.gradient_value = gradient[0]

        def activation_hook(a, b, activation):
            self.activation_value = activation

        conv_module.register_forward_hook(activation_hook)
        conv_module.register_backward_hook(gradient_hook)
        ##############################################################################
        # TODO: Implement GradCam as described in paper.                             #
        #                                                                            #
        # Compute a gradcam visualization using gc_model and convolution layer as    #
        # conv_module for images X_tensor and labels y_tensor.                       #
        #                                                                            #
        # Return:                                                                    #
        # If the activation map of the convolution layer we are using is (K, K) ,    #
        # student code should end with assigning a numpy of shape (N, K, K) to       #
        # a variable 'cam'. Instructor code would then take care of rescaling it     #
        # back                                                                       #
        ##############################################################################
        # output = gc_model(X_tensor)
        #
        # one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        # one_hot[0][y_tensor] = 1
        # one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        #
        # one_hot = torch.sum(one_hot * output)
        #
        # conv_module.zero_grad()
        # gc_model.zero_grad()
        # one_hot.backward(retain_graph=True)
        #
        # grads_val = self.gradient_value
        #
        # target = self.activation_value[-1]
        # target = target.cpu().data.numpy()[0, :]
        #
        # weights = np.mean(grads_val, axis=(2, 3))[0, :]
        # cam = np.zeros(target.shape[1:], dtype=np.float32)
        res = gc_model(X_tensor)
        #y_tensor = F.one_hot(y_tensor, 1000)
        #y_tensor = y_tensor.float()

        print(X_tensor.shape)
        print(res.shape)

        #loss = F.binary_cross_entropy_with_logits(res, y_tensor)
        loss = res.gather(1, y_tensor.view(-1, 1)).squeeze()
        loss = torch.sum(loss)
        loss.backward()

        N, C, H, W = self.gradient_value.shape
        neuron_importance = torch.sum(self.gradient_value, (2, 3)) / (H * W)
        neuron_importance = neuron_importance.view(N, C, 1, 1)

        cam = F.relu(torch.sum(neuron_importance * self.activation_value, 1)).detach().numpy()

        ##############################################################################
        #                             END OF YOUR CODE                               #
        ##############################################################################

        # Rescale GradCam output to fit image.
        cam_scaled = []
        for i in range(cam.shape[0]):
            cam_scaled.append(np.array(Image.fromarray(cam[i]).resize(X_tensor[i, 0, :, :].shape, Image.BICUBIC)))
        cam = np.array(cam_scaled)
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam
