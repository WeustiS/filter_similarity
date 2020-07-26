import os
from celluloid import Camera
from matplotlib import pyplot as plt
import matplotlib.image as mpimg

fig = plt.figure()
camera = Camera(fig)

for r, d, f in os.walk('.'):
    arr = [None] * 30
    for n in d:
        if 'default_full_l' in n:
            a, b = n.split('full_l')
            b = int(b)
            arr[b] = "./default_full_l"+str(b)+"/single.png"
    arr = [x for x in arr if x is not None]
    for path in arr:
        print(path)
        img = plt.imread(path)
        plt.imshow(img)
        camera.snap()
        plt.imshow(img)
        camera.snap()

print(camera._photos)
anim = camera.animate()
anim.save('single_layers_vgg.mp4')