import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Generator, self).__init__()

        """
        Encoder is responsible for taking in the cartoon image and down-sampling while retaining the core features 
        so it can later be reconstructed in the decoder. This is down using convolutional layers. 
        
        The Decoder takes the down-sampled features and up-samples it to using transpose layers. 
        
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),

        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            # Actually maps out the RGB channels
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh()  # Tanh activation for image generation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Exactly the same as the generator except it takes in the celeb images and generates cartoon images.
class InverseGenerator(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(Generator, self).__init__()

        """
        Encoder is responsible for taking in the cartoon image and down-sampling while retaining the core features 
        so it can later be reconstructed in the decoder. This is down using convolutional layers. 

        The Decoder takes the down-sampled features and up-samples it to using transpose layers. 

        """
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(),

            nn.Conv2d(hidden_dim, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(True),

            # Actually maps out the RGB channels
            nn.ConvTranspose2d(hidden_dim, 3, kernel_size=4, stride=2, padding=1, bias=False),

            nn.Tanh()  # Tanh activation for image generation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class Discriminator(nn.Module):
    def __init__(self, image_channels, hidden_dim):
        super(Discriminator, self).__init__()

        """
        We continue decreasing the Height x Width image until it becomes 1x1 then we flatten
        This is the only way i can think of for this to work since the dimension needs to be [batch_size]
        for the loss function. 

        We end with features hidden_dim * 8, but we then decrease to 1 to flatten it completely
        """
        self.main = nn.Sequential(

            nn.Conv2d(image_channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),

            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, x):
        output = self.main(x)

        return output