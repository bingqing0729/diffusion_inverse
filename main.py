from imagen_pytorch import Unet, Imagen, ImagenTrainer
from imagen_pytorch.data import Dataset, DataLoader
import numpy as np
import torch
# unets for unconditional imagen

unet = Unet(
    dim = 32,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 1,
    layer_attns = (False, False, False, True),
    layer_cross_attns = False
)

# imagen, which contains the unet above

imagen = Imagen(
    unets = unet,
    image_sizes = 64,
    timesteps = 1000
)

trainer = ImagenTrainer(
    imagen = imagen,
    split_valid_from_train = False # whether to split the validation dataset from the training
).cuda()

# instantiate your dataloader, which returns the necessary inputs to the DDPM as tuple in the order of images, text embeddings, then text masks. in this case, only images is returned as it is unconditional training

dataset = Dataset('../CurveVel_B/', image_size = 64)
trainer.add_train_dataset(dataset, batch_size = 16)

text_embeds = np.load('../CurveVel_B/data/data1.npy')
text_embeds = text_embeds[:2,:,:,:]
text_embeds = np.swapaxes(text_embeds, 2, 3)
text_embeds = text_embeds.reshape(text_embeds.shape[0], -1, text_embeds.shape[-1])
text_embeds = (text_embeds-np.min(text_embeds))/(np.max(text_embeds)-np.min(text_embeds))
text_embeds = torch.from_numpy(text_embeds)


trainer.load(f'./checkpoints/checkpoint-29.pt')
# working training loop

for i in range(1):
    loss = trainer.train_step(unet_number = 1, max_batch_size = 4)
    print(f'loss: {loss}')

    #if not (i % 500):
        #valid_loss = trainer.valid_step(unet_number = 1, max_batch_size = 4)
        #print(f'valid loss: {valid_loss}')

    if not (i % 1000) and trainer.is_main: # is_main makes sure this can run in distributed
        for j in range(5):
            images = trainer.sample(batch_size = 1, text_embeds = text_embeds, return_pil_images = True) # returns List[Image]
            images[1].save(f'./samples/sample_new-{j+5}.png')
        
        #trainer.save(f'./checkpoints/checkpoint-{i // 1000}.pt')
        