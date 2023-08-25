# CycleGan-Face

This is a unsupervised GAN that uses the CycleGan variant in order to convert Cartoon faces into Real People faces. 

# Architecture 

This uses the regular GAN architecture but in addition has 2 discriminators and 2 generators. There are 2 generators to convert cartoon faces into real faces and vice versa. Likewise, there is a discriminator to differentiate real cartoon faces vs fake cartoon faces and vice versa. 

Through training, the generator learns to take in the cartoon image, deconstruct its features, then reconstruct it into a real human face. 

# Results

After 60 hours of training the following results were achieved. 
![fake_real_222](https://github.com/durzal1/CycleGan-Face/assets/67489054/500a2a2e-f311-402d-9146-4360eda0ae4c)

The generator successfully creates human faces; however, they look vastly different than the cartoon images unfortunately. 

# Possible Improvements

In order to make this GAN better I believe following a conventional supervised approach would be much better than unsupervised. Moreover, the cartoon images have very exotic physiques that are unconventional for humans. 

Moreover, since the human dataset I used was of celebrities, the generator learned to make them look like celebrities instead of making them look like the cartoon images. This is also why there are some repeat results. 

# Conclusion

Overall, the GAN was able to create human faces but was unable to make them look like the cartoon images. 
