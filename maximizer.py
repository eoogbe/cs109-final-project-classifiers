class Maximizer(object):
    def __init__(self):
        self.__max = None
    
    def update(self, y, joint):
        if self.__max is None or self.__max < joint:
            self.y = y
            self.__max = joint
