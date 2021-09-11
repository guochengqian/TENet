def vis_numpy(x):
    import matplotlib.pyplot as plt
    plt.imshow(x, cmap='gray')
    plt.show()


def vis_tensor(tensor):
    """
    Input is tensor with length 4, with shape B, C, H, W
    """

    import torch
    from torchvision import utils
    import matplotlib.pyplot as plt

    grid = utils.make_grid(tensor)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    # vis_numpy(ndarr)
    plt.imshow(ndarr, cmap='gray')
    plt.show()


# useful for showing raw images
def raw_unpack(input):
    """
    Input is tensor with length 4, with shape B, C, H, W
    """
    import torch.nn as nn
    demo = nn.PixelShuffle(2)
    return demo(input)


# guidelines how to use:
# vis_tensor(raw_unpack(output[0:1, :4, :, :]))

