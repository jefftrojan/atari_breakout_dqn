import tkinter as tk
import random
import numpy as np

class BreakoutGame:
    def __init__(self, width=800, height=600):
        # Game window setup
        self.root = tk.Tk()
        self.root.title("Breakout Game")
        
        # Canvas setup
        self.width = width
        self.height = height
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg='black')
        self.canvas.pack()
        
        # Game objects
        self.paddle_width = 100
        self.paddle_height = 20
        self.paddle = self.canvas.create_rectangle(
            width//2 - self.paddle_width//2, 
            height - 50, 
            width//2 + self.paddle_width//2, 
            height - 30, 
            fill='blue'
        )
        
        self.ball = self.canvas.create_oval(
            width//2 - 10, height//2 - 10, 
            width//2 + 10, height//2 + 10, 
            fill='white'
        )
        
        # Ball movement
        self.ball_dx = random.choice([-5, 5])
        self.ball_dy = -5
        
        # Bricks
        self.bricks = []
        self.create_bricks()
        
        # Game state
        self.score = 0
        self.lives = 3
        self.score_display = self.canvas.create_text(
            50, 30, text=f"Score: {self.score}", 
            fill='white', font=('Arial', 16)
        )
        self.lives_display = self.canvas.create_text(
            width - 50, 30, text=f"Lives: {self.lives}", 
            fill='white', font=('Arial', 16)
        )
        
        # Keyboard bindings
        self.root.bind('<Left>', self.move_left)
        self.root.bind('<Right>', self.move_right)
        
        # AI Training attributes
        self.reset_count = 0
        self.max_steps = 1000
        self.current_step = 0

    def create_bricks(self):
        colors = ['red', 'green', 'blue', 'yellow', 'purple']
        for row in range(5):
            for col in range(10):
                x1 = col * (self.width // 10)
                y1 = row * 30
                x2 = x1 + (self.width // 10 - 5)
                y2 = y1 + 25
                brick = self.canvas.create_rectangle(
                    x1, y1, x2, y2, 
                    fill=colors[row % len(colors)]
                )
                self.bricks.append(brick)

    def move_left(self, event=None):
        # Move paddle left
        x1, _, x2, _ = self.canvas.coords(self.paddle)
        if x1 > 0:
            self.canvas.move(self.paddle, -20, 0)

    def move_right(self, event=None):
        # Move paddle right
        x1, _, x2, _ = self.canvas.coords(self.paddle)
        if x2 < self.width:
            self.canvas.move(self.paddle, 20, 0)

    def move_paddle_ai(self, direction):
        # AI paddle movement
        if direction == 'left':
            self.move_left()
        elif direction == 'right':
            self.move_right()

    def move_ball(self):
        # Move ball
        self.canvas.move(self.ball, self.ball_dx, self.ball_dy)
        ball_pos = self.canvas.coords(self.ball)
        
        # Wall collision
        if ball_pos[0] <= 0 or ball_pos[2] >= self.width:
            self.ball_dx *= -1
        
        # Ceiling collision
        if ball_pos[1] <= 0:
            self.ball_dy *= -1
        
        # Paddle collision
        paddle_pos = self.canvas.coords(self.paddle)
        if self.check_collision(ball_pos, paddle_pos):
            self.ball_dy *= -1
        
        # Brick collision
        for brick in self.bricks[:]:
            brick_pos = self.canvas.coords(brick)
            if self.check_collision(ball_pos, brick_pos):
                self.canvas.delete(brick)
                self.bricks.remove(brick)
                self.ball_dy *= -1
                
                # Update score
                self.score += 10
                self.canvas.itemconfig(
                    self.score_display, 
                    text=f"Score: {self.score}"
                )
                break
        
        # Ball below paddle
        if ball_pos[3] >= self.height:
            self.lives -= 1
            self.canvas.itemconfig(
                self.lives_display, 
                text=f"Lives: {self.lives}"
            )
            self.reset_ball()

    def check_collision(self, ball_pos, obj_pos):
        # Basic collision detection
        return (ball_pos[2] >= obj_pos[0] and 
                ball_pos[0] <= obj_pos[2] and 
                ball_pos[3] >= obj_pos[1] and 
                ball_pos[1] <= obj_pos[3])

    def reset_ball(self):
        # Reset ball to center
        self.canvas.coords(
            self.ball, 
            self.width//2 - 10, 
            self.height//2 - 10, 
            self.width//2 + 10, 
            self.height//2 + 10
        )
        self.ball_dx = random.choice([-5, 5])
        self.ball_dy = -5

    def step(self, action=None):
        # AI step function
        # Action: 0 = left, 1 = stay, 2 = right
        if action is not None:
            if action == 0:
                self.move_paddle_ai('left')
            elif action == 2:
                self.move_paddle_ai('right')
        
        # Move game elements
        self.move_ball()
        
        # Prepare state for AI
        state = self.get_state()
        
        # Check game over conditions
        done = self.lives <= 0 or len(self.bricks) == 0
        reward = self.calculate_reward(done)
        
        return state, reward, done

    def get_state(self):
        # Simplified state representation
        paddle_pos = self.canvas.coords(self.paddle)
        ball_pos = self.canvas.coords(self.ball)
        
        normalized_paddle = (paddle_pos[0] + paddle_pos[2]) / (2 * self.width)
        normalized_ball_x = (ball_pos[0] + ball_pos[2]) / (2 * self.width)
        normalized_ball_y = (ball_pos[1] + ball_pos[3]) / (2 * self.height)
        
        return [normalized_paddle, normalized_ball_x, normalized_ball_y]

    def calculate_reward(self, done):
        # Reward design
        reward = 0
        if done:
            reward = -100  # Penalty for losing
        else:
            reward = 1  # Small reward for staying alive