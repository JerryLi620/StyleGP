import random

class Line:
    def __init__(self, img_width, img_height):
        self.x = random.randint(0, int(img_width))
        self.y = random.randint(0, int(img_height))

        self._img_width = img_width
        self._img_height = img_height

    def mutate(self):
        new_tree = Line(self._img_width, self._img_height)
        x_shift = int(random.randint(-50, 50))
        y_shift = int(random.randint(-50, 50))
        new_tree.x, new_tree.y  = self.x+x_shift, self.y+y_shift
        return new_tree

    def crossover(self, other_tree):
        new_tree_1 = Line(self._img_width, self._img_height)
        new_tree_2 = Line(self._img_width, self._img_height)
        rand = random.random()

        # switch x
        if rand<0.5:
            new_tree_1.x, new_tree_1.y = self.x, other_tree.y
            new_tree_2.x, new_tree_2.y = other_tree.x, self.y
        #switch y
        else:
            new_tree_1.x, new_tree_1.y = other_tree.x, self.y
            new_tree_2.x, new_tree_2.y = self.x, other_tree.y
    
        return new_tree_1, new_tree_2
    

