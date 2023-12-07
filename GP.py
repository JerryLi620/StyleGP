import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImagePath
from Individual import Individual
import random
import math
from pandas import DataFrame
from feature_extractor import *
from content_extractor import *
import matplotlib.pyplot as plt

class GP:
    def __init__(self, content_image_name, style_image_name):
        content_image = Image.open(content_image_name).convert("RGB")
        self.style_image = Image.open(style_image_name).convert("RGB")
        self.style_gram = gram_matrix(extract_feature(self.style_image))
        # davidson image 
        # self.target_image = original_image.resize((160,120))

        # debugging 
        # self.target_image = original_image.resize((200,200))
        
        # mona lisa image
        # self.target_image = original_image.resize((176,203))
        
        # mona lisa twice as large (times 1.5)
        self.target_image = content_image.resize((256, 256))
        self.target_content = extract_content(self.target_image)
        # mona lisa twice as large (times 2.5)
        # self.target_image = original_image.resize((440,508))

        self.l, self.w = self.target_image.size
        
        self.target_image_array = self.to_array(self.target_image)


    def run_gp(self, pop_size, epochs):
        """
        Main driver of the genetic algorithm

        Keyword arguments:
        pop_size -- the population size for each generation
        epochs -- number of generations to run 

        Returns:
        fittest -- individual with the best fitness from final generation
        """

        data = {'epoch':[], 'fitness_content_estimate':[], 'fitness_style_estimate':[], 'crossover_used':[], 'pop_gen_used':[], 'im_size':[]}
        
        population = []

        # initialize starting population
        for i in range(pop_size):
            new_indiv = Individual(self.l, self.w)

            new_indiv.get_fitness(self.target_content, self.style_gram)
            
            population.append(new_indiv)

        for i in range(epochs):
            print("epoch:", i)
            new_pop = []

            # estimate for fitness of fittest individual from current epoch's population 
            fittest_style_estimate = float('inf')
            fittest_content_estimate = float('inf')
            # populate our new population
            while len(new_pop) < len(population):
                # select parents for crossover
                parent_one = self.tournament_select(population)
                parent_two = self.tournament_select(population)
                fittest_style_estimate = min(parent_one.style_loss, parent_two.style_loss, fittest_style_estimate)
                fittest_content_estimate = min(parent_one.content_loss, parent_two.content_loss, fittest_content_estimate)

                # probabilistically determine how child of both parents is created 
                rand = random.uniform(0, 1)
                        
                if rand < 0.3:
                    child = self.crossover(parent_one, parent_two)

                    while child == None:
                        parent_one = self.tournament_select(population)
                        parent_two = self.tournament_select(population)

                        child = self.crossover(parent_one, parent_two)

                elif rand <= 0.9:
                    child = self.crossover_2(parent_one, parent_two, 0.5)

                    while child == None:
                        parent_one = self.tournament_select(population)
                        parent_two = self.tournament_select(population)

                        child = self.crossover_2(parent_one, parent_two, 0.5)
                
                else:
                    child = self.mutate(parent_one)

                    while child == None:
                        parent_one = self.tournament_select(population)
                        child = self.mutate(parent_one)

                # add child to new population
                new_pop.append(child)

            # set population = new_pop
            population = new_pop
            
            # fitness data recording 
            if i % 1 == 0 or i == epochs - 1:
                data['epoch'].append(i)
                data['fitness_content_estimate'].append(fittest_content_estimate)
                data['fitness_style_estimate'].append(fittest_style_estimate)
                data['crossover_used'].append("crossover_1")
                data['pop_gen_used'].append("random_image_array_1")
                data['im_size'].append("(" + str(self.w) + "," + str(self.l) + ")")
            
            # save images on interval to see progress  
            if i % 1 == 0 or i == epochs - 1:

                print("Most fit individual in epoch " + str(i) +
                    " has content fitness: " + str(fittest_content_estimate) + 
                    " and style fitness: " + str(fittest_style_estimate))
                
                population.sort(key=lambda ind: ind.fitness)
                fittest = population[0]

                fittest.image.save("gif/fittest_" + str(i)+".png")
                
                data_df = DataFrame(data)
    
                data_df.to_csv("data_cross.csv")

        # save collected data to csv 
        data_df = DataFrame(data)
        data_df.to_csv("data_cross.csv")

        # fittest individual of the final population 
        population.sort(key=lambda ind: ind.fitness)
        fittest = population[0]

        return fittest


    def tournament_select(self, population, tournament_size=6):
        """
        Selects the most fit individual from a randomly sampled subset of the population

        Keyword arguments:
        population -- current generation's population
        tournament_size -- number of individuals randomly sampled to participate

        Returns:
        winner -- individual with the best fitness out of the tournament_size participants
        """

        # randomly sample participants
        indices = np.random.choice(len(population), tournament_size)
        random_subset = [population[i] for i in indices]

        winner = None

        # find individual with best fitness 
        for i in random_subset:
            if (winner == None):
                winner = i
            elif i.fitness < winner.fitness:
                winner = i

        return winner


    def crossover(self, ind1, ind2):
        """
        Performs 'blend' crossover given two parents and creates a child \
            It takes a weighted average of each parent and overlays them

        Keyword arguments:
        ind1 -- parent number 1
        ind2 -- parent number 2

        Returns:
        child or None -- child of the two parents if it is more fit than both parents
        """
        
        child = Individual(self.l, self.w)

        # random float between 0 and 1 
        blend_alpha = random.random()

        # if blend_alpha is 0.0, a copy of the first image is returned. 
        # If blend_alpha is 1.0, a copy of the second image is returned.
        # use a random blend_alpha \in (0,1) 
        child_image = Image.blend(ind1.image, ind2.image, blend_alpha).convert("RGB")
        child.image = child_image
        child.array = np.array(child_image)
        child.get_fitness(self.target_content, self.style_gram)

        # elitism 
        if child.fitness == min(ind1.fitness, ind2.fitness, child.fitness):
            return child

        return None

    
    def crossover_2(self, ind1, ind2, horizontal_prob):
        """
        Performs 'crossover point' crossover given two parents and creates a child \
            Randomly selects the crossover point to be either a row or column \
            Everything up until the crossover point is from parent 1, everything after is parent 2

        Keyword arguments:
        ind1 -- parent number 1
        ind2 -- parent number 2

        Returns:
        child or None -- child of the two parents if it is more fit than both parents
        """

        rand = random.random()

        # perform horizontal crossover point 
        if rand <= horizontal_prob:

            split_point = random.randint(1, self.w)
            
            first = np.ones((split_point, self.l))
            first = np.vstack((first, np.zeros((self.w-split_point, self.l))))

        # perform vertical crossover point 
        else:
            split_point = random.randint(1, self.l)
        
            first = np.ones((self.w, split_point))

            first = np.hstack((first, np.zeros((self.w, self.l-split_point))))
            
        second = 1 - first

        # Creates the 3 dimensional versions to perform the mutliplying across all color channels 
        first = np.dstack([first,first,first])
        second = np.dstack([second,second,second])

        # Multiply parent1 with first and multiply parent2 with second. Then simplay add them element wise and it should produce the crossover child.

        half_chromo_1 = np.multiply(first, ind1.array)
        half_chromo_2 = np.multiply(second, ind2.array)
        
        child_array = np.add(half_chromo_1, half_chromo_2)
        
        child = Individual(self.l, self.w)
        
        child.image = Image.fromarray(child_array.astype(np.uint8)).convert("RGB")
        child.array = child_array.astype(np.uint8)
        
        child.get_fitness(self.target_content, self.style_gram)

        # elitism 
        if child.fitness == min(ind1.fitness, ind2.fitness, child.fitness):
            return child

        return None


    def crossover_3(self, ind1, ind2):
        """
        Performs 'pixel-wise' crossover given two parents and creates a child \
            Each pixel is randomly selected from either parent 1 or parent 2

        Keyword arguments:
        ind1 -- parent number 1
        ind2 -- parent number 2

        Returns:
        child or None -- child of the two parents if it is more fit than both parents
        """
        first = np.random.randint(2, size=(self.w, self.l, 4))
        
        second = 1 - first
        
        half_chromo_1 = np.multiply(first, ind1.array)
        
        half_chromo_2 = np.multiply(second, ind2.array)
        
        child_array = np.add(half_chromo_1, half_chromo_2)
        
        child = Individual(self.l, self.w)
        
        child.image = Image.fromarray(child_array.astype(np.uint8)).convert("RGB")
        child.array = child_array.astype(np.uint8)
        
        child.get_fitness(self.target_content, self.style_gram)
        
        return child


    def mutate(self, ind):
        """
        Mutates an individual by superimposing a random number of randomly colored shapes

        Keyword arguments:
        ind -- individual to be mutated

        Returns:
        child -- the individual post mutation
        """

        iterations = random.randint(1, 3)
        region = random.randint(1,(self.l + self.w)//4)

        img = ind.image

        for i in range(iterations):
            num_points = random.randint(3, 6)
            region_x = random.randint(0, self.l)
            region_y = random.randint(0, self.w)

            xy = []
            for j in range(num_points):
                xy.append((random.randint(region_x - region, region_x + region),
                           random.randint(region_y - region, region_y + region)))

            img1 = ImageDraw.Draw(img)
            img1.polygon(xy, fill=ind.rand_color())

        child = Individual(ind.l, ind.w)
        child.image = img
        child.array = child.to_array(child.image)
        child.get_fitness(self.target_content, self.style_gram)

        return child 

        
    def mutate_2(self, ind):
        """
        Mutates an individual by selecting a random subset of pixels and altering their RGB values

        Keyword arguments:
        ind -- individual to be mutated

        Returns:
        child -- the individual post mutation
        """
        
        num_pix = 40
        
        for i in range(num_pix):
            x = random.randint(0, self.l-1)
            y = random.randint(0, self.w-1)
            z = random.randint(0, 3)
            
            ind.array[x][y][z] = ind.array[x][y][z] + random.randint(-10,10)
                
        ind.image = self.to_image(ind.array)
        ind.get_fitness(self.target_content, self.style_gram)

    def to_image(self, array):
        return Image.fromarray(array).convert("RGB")

    def to_array(self, image):
        return np.array(image)

# driver 
def main():
    gp = GP(r"dog.jpg", r"starry_night.jpg")

    fittest = gp.run_gp(100, 10000)
    plt.imshow(fittest.image)
    plt.show()

    # gp = GP(r"davidson2.png")

    # ind1 = Individual(100, 100)
    # plt.imshow(ind1.image)
    # plt.show()
    # ind2 = Individual(100, 100)
    # plt.imshow(ind2.image)
    # plt.show()

    # ind3 = gp.crossover_3(ind1, ind2)
    # plt.imshow(ind3.image)
    # plt.show()

if __name__ == "__main__":
    main()
