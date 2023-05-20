import common.game_constants as game_constants
import common.game_state as game_state
import pygame

class KeyboardController:
    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        keys = pygame.key.get_pressed()
        action = game_state.GameActions.No_action
        if keys[pygame.K_LEFT]:
            action = game_state.GameActions.Left
        if keys[pygame.K_RIGHT]:
            action = game_state.GameActions.Right
        if keys[pygame.K_UP]:
            action = game_state.GameActions.Up
        if keys[pygame.K_DOWN]:
            action = game_state.GameActions.Down
    
        return action

import numpy as np
import copy
class AIController:
### ------- You can make changes to this file from below this line --------------
    def __init__(self) -> None:
        # Add more lines to the constructor if you need
        # Initialize the Q-table with zeros
        self.epsilon = 1
        self.Q = np.zeros((game_constants.GAME_WIDTH, game_constants.GAME_HEIGHT, len(game_state.GameActions)))


    def GetAction(self, state:game_state.GameState) -> game_state.GameActions:
        # This function should select the best action at a given state
        if np.random.rand() < self.epsilon:
            # Explore
            action = np.random.choice(game_state.GameActions)
        else:
            # Exploit
            action = np.argmax(self.Q[int(state.PlayerEntity.entity.x), int(state.PlayerEntity.entity.y)])
        if not isinstance(action,game_state.GameActions):
            action = game_state.GameActions(action)
        # print(f"[LOG]: Action {game_state.GameActions(action)}")
        return action
        # A wrong example (just so you can compile and check)
        return game_state.GameActions.Right
    
    def TrainModel(self):
        # Complete this function
        # epochs = 10 # You might want to change the number of epochs
        # for _ in range(epochs):
        #     state = game_state.GameState()
        #     # Explore the state by updating it

        #     action = self.GetAction(state) # Select best action
        #     obs = state.Update(action) # obtain the observation made due to your action

        #     # You must complete this function by
        #     # training your model on the explored state,
        #     # using a suitable RL algorithm, and
        #     # by appropriately rewarding your model on 
        #     # the state that it lands

        #     pass
        # pass
        # Set the hyperparameters
        alpha = 0.55  # learning rate
        gamma = 0.99  # discount factor
        epochs = 100  # number of episodes

        for _ in range(epochs):
            state = game_state.GameState()
            x_old,y_old = state.PlayerEntity.entity.x, state.PlayerEntity.entity.y
            done = False
            reward = 0
            while not done:
                # Select the action with epsilon-greedy policy
                action = self.GetAction(state)

                # Update the state and get the observation and reward
                obs = state.Update(action)

                # Update the Q-table using Q-learning
                next_state = copy.deepcopy(state)
                next_obs = next_state.Update(game_state.GameActions.Right)
                # reward for distance b/w goal and player
                if np.sqrt((state.GoalLocation.y - y_old)**2 + (state.GoalLocation.x - x_old)**2) > np.sqrt((state.GoalLocation.y - state.PlayerEntity.entity.y)**2 + (state.GoalLocation.x - state.PlayerEntity.entity.x)**2):
                    reward += 25
                    state.PlayerEntity.velocity.x += 0.3
                    state.PlayerEntity.velocity.y += 0.3
                if np.sqrt((state.GoalLocation.y - y_old)**2 + (state.GoalLocation.x - x_old)**2) < np.sqrt((state.GoalLocation.y - state.PlayerEntity.entity.y)**2 + (state.GoalLocation.x - state.PlayerEntity.entity.x)**2):
                    reward += -50
                    state.PlayerEntity.velocity.x -= 0.3
                    state.PlayerEntity.velocity.y -= 0.3
                # reward for avoiding enemy
                # for enemy in state.EnemyCollection:
                #     if np.sqrt((enemy.entity.y - y_old)**2 + (enemy.entity.x - x_old)**2) < np.sqrt((enemy.entity.y - state.PlayerEntity.entity.y)**2 + (enemy.entity.x - state.PlayerEntity.entity.x)**2):
                #         reward += 0.5
                #     else:
                #         reward -= 15
                # +ve reward
                if next_obs == game_state.GameObservation.Reached_Goal:
                    if self.epsilon > 0.3:
                        self.epsilon -= 0.1
                    # if alpha > 0.25:
                    #     alpha -= 0.1
                    reward += 100
                    done = True
                # -ve reward
                elif next_obs == game_state.GameObservation.Enemy_Attacked:
                    reward += -80
                    done = True
                else:
                    reward += 0
                # print(f"[LOG] Distance {np.linalg.norm((state.PlayerEntity.entity.x,state.PlayerEntity.entity.y)-(state.GoalLocation.x,state.GoalLocation.y))}, {np.linalg.norm((next_state.PlayerEntity.entity.x,state.PlayerEntity.entity.y)-(next_state.GoalLocation.x,state.GoalLocation.y))}")
                # print(f"[LOG] State PlayerEntity x: {state.PlayerEntity.entity.x}, State PlayerEntity y: {state.PlayerEntity.entity.y}, Action: {action}")
                # print(f"[LOG] Next State PlayerEntity x: {int(next_state.PlayerEntity.entity.x)}, Next State PlayerEntity y: {int(next_state.PlayerEntity.entity.y)}")
                self.Q[int(state.PlayerEntity.entity.x)+1, int(state.PlayerEntity.entity.y)+1, action.value] = (1 - alpha) * self.Q[int(state.PlayerEntity.entity.x), int(state.PlayerEntity.entity.y), action.value] + alpha * (reward + gamma * np.max(self.Q[int(next_state.PlayerEntity.entity.x), int(next_state.PlayerEntity.entity.y)]))
                state = next_state


### ------- You can make changes to this file from above this line --------------

    # This is a custom Evaluation function. You should not change this function
    # You can add other methods, or other functions to perform evaluation for
    # yourself. However, this evalution function will be used to evaluate your model
    def EvaluateModel(self):
        attacked = 0
        reached_goal = 0
        state = game_state.GameState()
        for _ in range(100000):
            action = self.GetAction(state)
            obs = state.Update(action)
            if(obs==game_state.GameObservation.Enemy_Attacked):
                attacked += 1
            elif(obs==game_state.GameObservation.Reached_Goal):
                reached_goal += 1
        return (attacked, reached_goal)
