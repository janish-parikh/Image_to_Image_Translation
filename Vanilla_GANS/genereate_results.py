from networks import Generator
import matplotlib.pyplot as plt
import torch,os
from torchvision.utils import save_image
for i in range(0, 51):
    model = Generator()
    model.load_state_dict(torch.load(f"./modelbak/generator_epoch{i}.pth"))
    z = torch.randn(4, 3, 256, 256, device="cpu")
    gen_imgs = model(z)
    # print(gen_imgs.shape)
    os.makedirs("results", exist_ok=True)
    save_image(
                    gen_imgs.data[:4],
                    f"./results/epoch{i+1}.jpg",
                    nrow=2,
                    normalize=True,
                )
