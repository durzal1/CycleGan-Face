import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from dataset import *
from tqdm import tqdm
from model import *
from utils import *
import torch

# Constants
HEIGHT = 64
WIDTH = 64
cartoon_path = 'cartoonset100k'
celeb_path = 'img_align_celeba'
DEVICE = 'cuda'

# Hyper Parameters
BATCH_SIZE = 64
NUM_EPOCHS = 100
LEARNING_RATE1 = 0.0001
LEARNING_RATE2 = 0.002
LEARNING_RATE3 = 0.002
HIDDEN_SIZE = 64

# Custom Dataset
dataset = CustomDataset(HEIGHT, WIDTH, cartoon_path, celeb_path)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Initialize models
discriminator_cartoon = Discriminator(3, HIDDEN_SIZE).to(DEVICE)
discriminator_real = Discriminator(3,HIDDEN_SIZE).to(DEVICE)

generator = Generator(3, HIDDEN_SIZE).to(DEVICE)
inverse_generator = InverseGenerator(3, HIDDEN_SIZE).to(DEVICE)

# Load the generator and discriminator models
# generator.load_state_dict(torch.load('generator.pth'))
# discriminator.load_state_dict(torch.load('discriminator.pth'))

# Initialize optimizers
optimizer_D_cartoon = torch.optim.Adam(discriminator_cartoon.parameters(), lr=LEARNING_RATE1)
optimizer_D_real = torch.optim.Adam(discriminator_real.parameters(), lr=LEARNING_RATE1)

optimizer_G = torch.optim.Adam(list(generator.parameters()) + list(inverse_generator.parameters()), lr=LEARNING_RATE2)

# Binary cross-entropy loss
criterion = nn.BCELoss()

print(f'The Discriminator has {count_parameters(discriminator_real):,} trainable parameters')
print(f'The Generator has {count_parameters(generator):,} trainable parameters')


# Training loop
for epoch in range(NUM_EPOCHS):
    loop = tqdm(train_loader, leave=True)

    discriminator_loss = 0
    generator_loss = 0

    for batch_idx, (cartoon_images, real_images) in enumerate(loop):
        cartoon_images = cartoon_images.to(DEVICE)
        real_images = real_images.to(DEVICE)

        batch_size = cartoon_images.size(0)

        real_labels = torch.ones(batch_size, 1)
        fake_labels = torch.zeros(batch_size, 1)

        # Training the real discriminator (Feed in real)
        optimizer_D_real.zero_grad()

        real_outputs = discriminator_real(real_images)
        fake_outputs = discriminator_real(generator(cartoon_images).detach())

        # Differentiate the real images vs the fake ones
        real_discriminator_loss = criterion(real_outputs, real_labels) + \
                                              criterion(fake_outputs, fake_labels)

        real_discriminator_loss.backward()
        optimizer_D_real.step()

        # Train the cartoon discriminator (Feed in cartoon)
        optimizer_D_cartoon.zero_grad()

        real_outputs = discriminator_cartoon(cartoon_images)
        fake_outputs = discriminator_cartoon(inverse_generator(real_images).detach())

        # Differentiate the real cartoon images vs the fake ones
        cartoon_discriminator_loss = criterion(real_outputs, real_labels) + \
                                              criterion(fake_outputs, fake_labels)

        cartoon_discriminator_loss.backward()
        optimizer_D_cartoon.step()

        ## Training the Generators (we only use one optimizer)
        optimizer_G.zero_grad()

        # Gen creates fake image and tries to trick discriminator
        fake_outputs_real = discriminator_real(generator(cartoon_images))
        real_generator_loss = criterion(fake_outputs_real, real_labels)

        # Inverse Gen creates fake cartoon and tries to trick discriminator
        fake_outputs_cartoon = discriminator_cartoon(inverse_generator(real_images))
        cartoon_generator_loss = criterion(fake_outputs_cartoon, real_labels)



        # Cycle consistency loss

        reconstructed_cartoon_images = inverse_generator(generator(cartoon_images))
        cycle_consistency_loss = torch.mean(torch.abs(reconstructed_cartoon_images - cartoon_images))

        total_gen_loss = cycle_consistency_loss + real_generator_loss + cartoon_generator_loss
        total_gen_loss.backward()
        optimizer_G.step()

        # Save images occasionally
        if epoch % 1 == 0 and batch_idx == 0:
            SavePNG(generator(cartoon_images), inverse_generator(real_images), epoch)

        discriminator_loss += real_discriminator_loss + cartoon_discriminator_loss
        generator_loss += total_gen_loss

    average_generator_loss = generator_loss / len(train_loader)
    average_discriminator_loss = discriminator_loss / len(train_loader)

    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Discriminator Loss: {average_discriminator_loss:.4f}"
          f", Generator Loss: {average_generator_loss:.4f}")

    # Occasionally save the model
    if epoch % 3 == 0:
        torch.save(generator.state_dict(), 'generator.pth')
        torch.save(inverse_generator.state_dict(), 'inverse_generator.pth')
        torch.save(discriminator_real.state_dict(), 'discriminator_real.pth')
        torch.save(discriminator_cartoon.state_dict(), 'discriminator_cartoon.pth')