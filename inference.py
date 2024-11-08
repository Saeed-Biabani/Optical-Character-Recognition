from src.tokenizer import loadTokenizer
from src.transforms import Transforms
import matplotlib.pyplot as plt
from src.misc import loadImage
from hazm import Normalizer
from src.nn import TRnet
import torch
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, metadata = TRnet.from_pretrained('model-new.pth')
model = model.to(device)

tokenizer = loadTokenizer(metadata.vocab_path)

if len(sys.argv) == 2:
    _, image_file_pth = sys.argv
    image = Transforms((32, 512))(
        loadImage(
            image_file_pth
        )
    )

plt.imshow(image[0], cmap = 'gray')
plt.savefig('image_sample.png')

generatd = model.read_image(
    pixel_values = image[None, ...]
)
gen_text = tokenizer.decode(generatd[0], ignore_special = True)


normalizer = Normalizer()
norm_gen = normalizer.normalize(gen_text)

print(f"Prediction:\n\t{norm_gen}")