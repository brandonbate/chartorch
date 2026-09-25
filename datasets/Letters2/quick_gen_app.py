import torch
import numpy as np
import string
import random
from PIL import Image, ImageDraw, ImageFont

images = []
labels = []

fonts = [ImageFont.truetype("LiberationSans-Bold.ttf", size=26),
        ImageFont.truetype("LiberationSans-BoldItalic.ttf", size=26),
        ImageFont.truetype("LiberationSans-Italic.ttf", size=26),
        ImageFont.truetype("LiberationSans-Regular.ttf", size=26),
        ImageFont.truetype("LiberationSerif-Bold.ttf", size=26),
        ImageFont.truetype("LiberationSerif-BoldItalic.ttf", size=26),
        ImageFont.truetype("LiberationSerif-Italic.ttf", size=26),
        ImageFont.truetype("LiberationSerif-Regular.ttf", size=26),
        ImageFont.truetype("LiberationMono-Bold.ttf", size=26),
        ImageFont.truetype("LiberationMono-BoldItalic.ttf", size=26),
        ImageFont.truetype("LiberationMono-Italic.ttf", size=26),
        ImageFont.truetype("LiberationMono-Regular.ttf", size=26)];


for font in fonts:

    for i, letter in enumerate(string.ascii_uppercase):

        for j in range(10):
            image = Image.new('L', (30, 30), color=0)
            draw = ImageDraw.Draw(image)
            x =random.randint(1, 10)
            y =random.randint(-5, 5)

            draw.text((x, y), letter, fill=255, font=font)

            tensor_img = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
            tensor_img = tensor_img.unsqueeze(0) 

            images.append(tensor_img)
            labels.append(i)

x = torch.stack(images)
y = torch.tensor(labels, dtype=torch.long)

# Save the tensors to a file
torch.save({'images': x, 'labels': y}, 'letters_dataset.pt')

# Test data
images = []
labels = []

for font in fonts:

    for i, letter in enumerate(string.ascii_uppercase):

        image = Image.new('L', (30, 30), color=0)
        draw = ImageDraw.Draw(image)
        x =random.randint(1, 10)
        y =random.randint(-5, 5)

        draw.text((x, y), letter, fill=255, font=font)

        tensor_img = torch.tensor(np.array(image), dtype=torch.float32) / 255.0
        tensor_img = tensor_img.unsqueeze(0) 

        images.append(tensor_img)
        labels.append(i)

x = torch.stack(images)
y = torch.tensor(labels, dtype=torch.long)

# Save the tensors to a file
torch.save({'images': x, 'labels': y}, 'test_letters_dataset.pt')


