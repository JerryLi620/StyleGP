import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImagePath
from PIL import ImageDraw
from IPython.display import display
import colour
from feature_extractor import *
from content_extractor import *
import matplotlib.pyplot as plt

import random
import math

# from IPython.display import Image

class Individual:
    def __init__(self, l, w):
        self.l = l
        self.w = w
        self.fitness = float('inf')
        self.style_loss = float('inf')
        self.content_loss = float('inf')
        self.array = None
        self.image = None
        coinflip = random.randint(1, 4)
        # if coinflip == 3:
        #     self.create_random_image_array_2()
        # else:
        self.create_random_image_array()

    def rand_color(self):
        return "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])

    def create_one_color(self):
        self.image = Image.new(mode="RGB", size=(self.l, self.w), color=self.rand_color())

    def create_random_image_array(self):

        # number of polygons to add to image
        # the higher this is the higher stochasticity and potential for detail we have 
        iterations = random.randint(3, 6)

        region = (self.l + self.w)//8

        img = Image.new("RGB", (self.l, self.w), self.rand_color())

        #number of points for each polygon
        for i in range(iterations):
            num_points = random.randint(3, 6)

            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))

            # xy = [
            #     ((math.cos(th) + 1) * 90,
            #      (math.sin(th) + 1) * 60)
            #     for th in [i * (2 * math.pi) / num_points for i in range(num_points)]
            # ]

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=self.rand_color())

        self.image = img
        self.array = self.to_array(img)
        
    def create_random_image_array_2(self):
        self.array = np.random.randint(low = 0, high = 255, size = (self.l, self.w, 4))
        
        self.array = self.array.astype('uint8')

        self.image = Image.fromarray(self.array.astype('uint8'))

    def add_shape(self):
        iterations = random.randint(1, 1)

        region = random.randint(1,(self.l + self.w)//4)

        img = self.image

        for i in range(iterations):
            num_points = random.randint(3, 6)

            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=self.rand_color())

        self.image = img
        self.array = self.to_array(img)

    def to_image(self):
        im = Image.fromarray(self.array)
        im.show()

    def to_array(self, image):
        return np.array(image)


# Try this for later 
# PIL.ImageChops.difference(image1, image2)[source]
# Returns the absolute value of the pixel-by-pixel difference between the two images.
    def get_fitness(self, target_content, style_gram):
        content = extract_content(self.image)
        self.content_loss = torch.mean(np.abs(target_content - content))
        feature = extract_feature(self.image)
        gram = gram_matrix(feature)
        self.style_loss = 0
        for i in range(len(gram)):
            self.style_loss += torch.mean(np.abs(style_gram[i] - gram[i]))* 1e3
        # print(style_loss, content_loss)
        total_loss = self.content_loss + self.style_loss 
        self.fitness = total_loss
        return total_loss
    
    def get_fitness_euclidean(self, target):
        diff_array = np.subtract(np.array(target), self.array)
        self.fitness = np.mean(np.absolute(diff_array))

# ind = Individual(175,175)
# display(ind.image)
# plt.imshow(ind.image)
# plt.show()