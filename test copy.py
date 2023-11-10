from flask import Flask, jsonify
import torch
from PIL import Image
import cv2
import numpy as np
import os
import pandas as pd
import torchvision.transforms as transforms

app = Flask(__name__)

GOOD = ['5AA', '4AA', '3AA', '5AB', '4AB']
FAIR = ['3AB', '6AA', '6AB', '5BA', '4BA', '3BA', '5BB', '4BB', '3BB', '6BA', '6BB', '2AA', '2AB', '2BA', '2BB']
BAD = ['5AC', '4AC', '3AC', '6AC', '5BC', '4BC', '3BC', '6BC', '2A-', '2BC', '2B-', '5CA', '4CA', '3CA', '6CA', '2CA', '2C-', '2-A', '5CB', '4CB', '3CB', '6CB', '2CB', '2-B', '5CC', '4CC', '3CC', '6CC', '2CC', '2-C', '2--', '1AA', '1AB', '1A-', '1BA', '1BB', '1B-', '1-A', '1-B', '1AC', '1CA', '1BC', '1CB', '1CC', '1C-','1-C', '1--']
fulllist = {}
map_location = torch.device('cpu')
for idx, i in enumerate(GOOD):
    fulllist[i] = 101.5 - 4 * (idx + 1.5)
for idx, i in enumerate(FAIR):
    fulllist[i] = 76 - float(25 / (len(FAIR) - idx))
for idx, i in enumerate(BAD):
    fulllist[i] = 51 - float(50 / (len(BAD) - idx))
path = 'C:/Users/jibin/Downloads/predict_grades/Final/'
net = torch.load(path+'Best_expansion.pth', map_location=torch.device('cpu'))
net1 = torch.load(path+'Best_icm.pth', map_location=torch.device('cpu'))
net2 = torch.load(path+'Best_te.pth', map_location=torch.device('cpu'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
net = net.to(device)
net1 = net1.to(device)
net2 = net2.to(device)
net.eval()
net1.eval()
net2.eval()


def transform():
    transform = transforms.Compose([
        # transforms.Resize((512,512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0],  # 485,0.456,0.406],
                             std=[1.0, 1.0, 1.0], ),  # 0.229,0.224,0.225
    ])
    return transform


dict1 = {0: '-', 1: 'A', 2: 'B', 3: 'C'}
dict2 = {0: '-', 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}


@app.route('/upload_process')
def process_images():
    result = []
    rgb_image = np.zeros((512, 512, 3), dtype=np.uint8)
    for p, d, f in os.walk('C:/Users/jibin/Downloads/craft/'):
        for f1 in f:
            im = cv2.imread(f1)
            if im is not None:
                gray_image = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                gray_image = cv2.resize(gray_image, (512, 512))

                rgb_image[:, :, 0] = gray_image
                rgb_image[:, :, 1] = gray_image
                rgb_image[:, :, 2] = gray_image
                im2 = Image.fromarray(rgb_image)

                im2 = transform()(im2)
                im2 = im2[None, :, :, :]
                im2 = im2.to(device)
                torch.cuda.empty_cache()
                out = net(im2)
                out1 = net1(im2)
                out2 = net2(im2)
                del(im2)
                row = out.detach().cpu().numpy()
                row1 = out1.detach().cpu().numpy()
                row2 = out2.detach().cpu().numpy()
                del(out1)
                del(out2)
                del(out)
                newdat = np.argmax(row)
                newdat1 = np.argmax(row1)
                newdat2 = np.argmax(row2)
                fullgrade = str(dict2[newdat]) + str(dict1[newdat1]) + str(dict1[newdat2])
                fullgrade_value = str(dict2[newdat]) + str(dict1[newdat1]) + str(dict1[newdat2])
                if str(fullgrade) in GOOD:
                    quality = 'Best quality embryo viable for Freezing'
                elif str(fullgrade) in FAIR:
                    quality = 'Fair quality embryo, blastocyst developing'
                else:
                    quality = 'Poor quality embryo'

                result.append({
                    "class": quality,
                    "filename": f1,
                    "img": "https://lab.genesysailabs.com/uploads/" + f1,
                    "percentage": str(fulllist[fullgrade])
                })

    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
