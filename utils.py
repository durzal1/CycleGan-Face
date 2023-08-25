import os

from matplotlib import pyplot as plt

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def SavePNG(cartoon, fake_real, fake_cartoon, epoch):
    # Convert pixel values from [-1, 1] to [0, 1] for imshow
    fake_real = (fake_real + 1) / 2
    fake_cartoon = (fake_cartoon + 1) / 2
    cartoon = (cartoon + 1) / 2

    # Fake Real
    fig, axs = plt.subplots(nrows=8, ncols=2, figsize=(64, 64))
    axs = axs.ravel()

    add = 0
    for idx in range(8):
        image_np = fake_real[idx].permute(1, 2, 0).detach().cpu().numpy()
        real_np = cartoon[idx].permute(1, 2, 0).detach().cpu().numpy()

        axs[idx + add].imshow(real_np)
        add += 1
        axs[idx + add].imshow(image_np)

    # for idx, image in enumerate(fake_real):
    #     image_np = image.permute(1, 2, 0).detach().cpu().numpy()
    #
    #     axs[idx].imshow(image_np)
    #     axs[idx].axis('off')

    # Save the grid as a PNG file
    grid_filename = os.path.join("Created", f"fake_real_{epoch}.png")
    plt.tight_layout()
    plt.savefig(grid_filename)
    plt.close()

    # Fake Cartoon
    fig, axs = plt.subplots(nrows=8, ncols=8, figsize=(10, 10))
    axs = axs.ravel()

    for idx, image in enumerate(fake_cartoon):
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        real_np = cartoon[idx].permute(1, 2, 0).detach().cpu().numpy()
        axs[idx].imshow(image_np)
        axs[idx].axis('off')

    # Save the grid as a PNG file
    grid_filename = os.path.join("Created", f"fake_cartoon_{epoch}.png")
    plt.tight_layout()
    plt.savefig(grid_filename)
    plt.close()
