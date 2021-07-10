import pygame
import random
from enum import Enum
import matplotlib.pyplot as plt
from collections import namedtuple
import numpy as np
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
from copy import deepcopy

pygame.init()
# font = pygame.font.Font("./arial.ttf", 25)

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4
    
Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200,0,0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
GREEN = (0, 255, 0)
BLACK = (0,0,0)

BLOCK_SIZE = 10
SPEED = 20


class SnakeGame:
    
    def __init__(self, w=80, h=80 , max_reward = 1,food_nb = 1 , early_stopping_factor = 10 ,gray_scale = False, isdisplayed = False, use_images = True  , image_reduction = 1 ):
        self.w = w
        self.h = h
        self.use_images = use_images        
        self.gray_scale = gray_scale
        self.isdisplayed = isdisplayed 
        self.reduction = image_reduction
        self.food_nb = food_nb
        if isdisplayed:
            pygame.display.set_caption('Snake')
            self.display = pygame.display.set_mode((self.w, self.h))
            self.clock = pygame.time.Clock()
        self.max_reward = max_reward
        self.early_stopping_factor = early_stopping_factor
        self.actions = np.arange(3)
        if gray_scale and use_images:
            self.observation_space =(1,w//self.reduction,h//self.reduction)
        elif not gray_scale and use_images:
            self.observation_space =(3,w//self.reduction,h//self.reduction)
        else:
            self.observation_space = (11,1)

    
    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w/2, self.h/2)
        self.snake = [self.head, 
                      Point(self.head.x+BLOCK_SIZE, self.head.y),
                      Point(self.head.x+(2*BLOCK_SIZE), self.head.y),
                      Point(self.head.x+(3*BLOCK_SIZE), self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food('start')
        self.frame_iteration = 0
        if self.isdisplayed:
            self._update_ui()
        return self._get_state()

        
    def _place_food(self,idx):
        x = [random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]
        y = [random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]

        if idx == 'start':
            self.food = [Point(x[i], y[i]) for i in range(self.food_nb)] 

        else: 
            self.food[idx] = Point(x[0], y[0]) 
        i = 0

        while (bool(set(self.food) & set(self.snake)) or len(set(self.food)) !=self.food_nb) and (i<10):
            i += 1 
            x = [random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]
            y = [random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]
            if idx == 'start':
                
                self.food = [Point(x[i], y[i]) for i in range(self.food_nb)] 
            else: 
                
                self.food[idx] = Point(x[0], y[0])  
            

        if i>=10:
            while (bool(set(self.food) & set(self.snake)) or len(set(self.food)) !=self.food_nb):
                x = [random.randint(0, (self.w-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]
                y = [random.randint(0, (self.h-BLOCK_SIZE )//BLOCK_SIZE )*BLOCK_SIZE for i in range(self.food_nb)]
                self.food = [Point(x[i], y[i]) for i in range(self.food_nb)]

    def set_display(self,boolean):
        self.isdisplayed = boolean


    def _get_state(self):
        if self.use_images:

            img = np.zeros((3, self.observation_space[1]*self.reduction,self.observation_space[2]*self.reduction ))        
            for i,pt in enumerate(self.snake):
                if i == 0:
                    img[1,int(pt.x):int(pt.x)+BLOCK_SIZE,int(pt.y):int(pt.y)+BLOCK_SIZE] = 1.
                else:
                    img[2,int(pt.x):int(pt.x)+BLOCK_SIZE,int(pt.y):int(pt.y)+BLOCK_SIZE] = 1.
            for i,pt in enumerate(self.food):
                img[0,int(pt.x):int(pt.x)+BLOCK_SIZE,int(pt.y):int(pt.y)+BLOCK_SIZE] = 1.
                #pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
                #text = font.render("Score: " + str(self.score), True, WHITE)
                #self.display.blit(text, [0, 0])
                # pygame.display.flip()
                
                # img = np.fromstring(pygame.image.tostring(self.display, "RGB"), np.uint8).reshape(*self.display.get_size(),3).copy()
                #img = Image.fromarray(np.uint8(img)).astype(np.float32)
            #img = img.transpose(1,2,0)
            img = rescale(img.transpose(1,2,0), 1/self.reduction, preserve_range=True, anti_aliasing=False).transpose(2,1,0)
            if self.gray_scale :
                img = np.dot([0.2989, 0.5870, 0.1140],img)
                img =  np.expand_dims(img, axis=0)
                
            
                #img = img.transpose(2,0,1)
            
            return img
        else:
            head = self.head
            point_l = Point(head.x - 20, head.y)
            point_r = Point(head.x + 20, head.y)
            point_u = Point(head.x, head.y - 20)
            point_d = Point(head.x, head.y + 20)
            
            dir_l = self.direction == Direction.LEFT
            dir_r = self.direction == Direction.RIGHT
            dir_u = self.direction == Direction.UP
            dir_d = self.direction == Direction.DOWN

            state = [
                # Danger straight
                (dir_r and self.is_collision(point_r)) or 
                (dir_l and self.is_collision(point_l)) or 
                (dir_u and self.is_collision(point_u)) or 
                (dir_d and self.is_collision(point_d)),

                # Danger right
                (dir_u and self.is_collision(point_r)) or 
                (dir_d and self.is_collision(point_l)) or 
                (dir_l and self.is_collision(point_u)) or 
                (dir_r and self.is_collision(point_d)),

                # Danger left
                (dir_d and self.is_collision(point_r)) or 
                (dir_u and self.is_collision(point_l)) or 
                (dir_r and self.is_collision(point_u)) or 
                (dir_l and self.is_collision(point_d)),
                
                # Move direction
                dir_l,
                dir_r,
                dir_u,
                dir_d,
                
                # Food location 
                self.food[0].x < self.head.x,  # food left
                self.food[0].x > self.head.x,  # food right
                self.food[0].y < self.head.y,  # food up
                self.food[0].y > self.head.y  # food down
                ]

            return np.array(state, dtype=int)


    def copy(self):
        self.isdisplayed = False 
        copyobj = SnakeGame()
        for name, attr in self.__dict__.items():
            if hasattr(attr, 'copy') and callable(getattr(attr, 'copy')):
                copyobj.__dict__[name] = attr.copy()
            else:
                copyobj.__dict__[name] = deepcopy(attr)
        return copyobj

    def get_discreet(self):
        head = self.head
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            self.food[0].x < self.head.x,  # food left
            self.food[0].x > self.head.x,  # food right
            self.food[0].y < self.head.y,  # food up
            self.food[0].y > self.head.y  # food down
            ]

        return np.array(state, dtype=int)

    
    def step(self , action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            
        self._move(action) 
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > self.early_stopping_factor*len(self.snake):
            game_over = True
            reward = - self.max_reward
            state = self._get_state()
            return state , reward , game_over, self.score
            
        if self.head in self.food:
            self.score += 1
            reward = self.max_reward
            idx = self.food.index(self.head)
            self._place_food(idx)
        else:
            self.snake.pop()

        if self.isdisplayed:
            self._update_ui()
            self.clock.tick(SPEED)
        state = self._get_state()
        
        return state , reward , game_over, self.score
    
    def is_collision(self, pt=None):
        if pt is None:
            pt =self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False


        
    def _update_ui(self):
        self.display.fill(BLACK)
        
        for i,pt in enumerate(self.snake):
            if i == 0:
                pygame.draw.rect(self.display, GREEN, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
                
            else:
                pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))

        for i,pt in enumerate(self.food): 
            pygame.draw.rect(self.display, RED, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        
        #pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        #text = font.render("Score: " + str(self.score), True, WHITE)
        #self.display.blit(text, [0, 0])
        pygame.display.flip()
        
    def _move(self, action):

        #action = [straight, right , left]

        clock_wise = [Direction.RIGHT, Direction.DOWN , Direction.LEFT , Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, 0):
            new_direction = clock_wise[idx]
        if np.array_equal(action, 1):
            idx = (idx+1)%4
            new_direction = clock_wise[idx]
        if np.array_equal(action, 2):
            idx = (idx-1)%4
            new_direction = clock_wise[idx]
        x = self.head.x
        y = self.head.y
        self.direction = new_direction
        

        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE
            
        self.head = Point(x, y)

            