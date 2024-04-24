# -*- coding: utf-8 -*-
# Introdução ao Aprendizado por Reforço - PPGEE
# Prof. Armando Alves Neto
########################################################################
try:
    import gymnasium as gym
    from gymnasium import spaces
except:
    import gym
    from gym import spaces
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
from functools import partial

NACTIONS = 9
FONTSIZE = 12
MAX_STEPS = 100

########################################
# classe do mapa
########################################
class Maze(gym.Env):
    ########################################
    # construtor
    def __init__(self, xlim=np.array([0.0, 10.0]), ylim=np.array([0.0, 10.0]), res=0.4, img='cave.png', alvo=np.array([9.5, 9.5])):

        # salva o tamanho geometrico da imagem em metros
        self.xlim = xlim
        self.ylim = ylim

        # resolucao
        self.res = res

        ns = int(np.max([np.abs(np.diff(self.xlim)), np.abs(np.diff(self.ylim))])/res)
        self.num_states = [ns, ns]
        
        # espaco de atuacao
        self.action_space = spaces.Discrete(NACTIONS)

        # cria mapa
        self.init2D(img)

        # converte estados continuos em discretos
        lower_bounds = [self.xlim[0], self.ylim[0]]
        upper_bounds = [self.xlim[1], self.ylim[1]]
        self.get_state = partial(self.obs_to_state, self.num_states, lower_bounds, upper_bounds)

        # alvo
        self.alvo = alvo
        
        
        # Configurações do rádio
        self.PT = 28
        self.At = 16
        self.Ar = 6
        
        self.radio_reward = 5

        self.antenas = [(3,3), (6,6)]
        # self.antenas = [(8,1), (1,8)]

        self.dBm_lim = [-35, -100]
        self.d0 = 10
        self.L0 = 124
        self.n = 2
        
        self.radio_best_signal_range = -40
        self.average_radio_signal = -60
        self.bad_radio_signal = -80
        
        self.x, self.y, self.dbm, self.rgb = [], [], [], []
        
        # Limits of the marker array color
        self.green_max = 1
        self.green_min = 0

        self.limit_max_dBm = -15
        self.radio_max_dBm = self.limit_max_dBm + 100

        self.limit_low_dBm = -80
        self.radio_min_dBm = self.limit_low_dBm + 100

        # Ranges of the radio readings 
        self.OldRange_dBm = 0
        self.NewRange = 0
        self.get_range()
        self.radio_beheaviour()
        

    ########################################
    # seed
    ########################################
    def seed(self, rnd_seed = None):
        np.random.seed(rnd_seed)
        return [rnd_seed]

    ########################################
    # reset
    ########################################
    def reset(self):

        # numero de passos
        self.steps = 0

        # posicao aleatória
        self.p = self.getRand()

        return self.get_state(self.p)

    ########################################
    # converte acão para direção
    def actionU(self, action):
        
        # action 0 faz ficar parado
        if action == 0:
            r = 0.0
        else:
            r = self.res
        
        action -= 1
        th = np.linspace(0.0, 2.0*np.pi, NACTIONS)[:-1]
        
        return r*np.array([np.cos(th[action]), np.sin(th[action])])
        
    ########################################
    # step -> new_observation, reward, done, info = env.step(action)
    def step(self, action):

        # novo passo
        self.steps += 1
        
        # seleciona acao
        u = self.actionU(action)

        # proximo estado
        nextp = self.p + u

        # fora dos limites (norte, sul, leste, oeste)
        if ( (self.xlim[0] <= nextp[0] < self.xlim[1]) and (self.ylim[0] <= nextp[1] < self.ylim[1]) ):
            self.p = nextp
         
        # reward
        reward = self.getReward()
        
        # estado terminal?
        done = self.terminal()

        # retorna
        return self.get_state(self.p), reward, done, {}

    ########################################
    # função de reforço
    def getReward(self):
        
        # reward
        reward = 0.0
        
        # colisao
        if self.collision(self.p):
            reward -= MAX_STEPS/2.0
            
        # chegou no alvo
        if np.linalg.norm(self.p - self.alvo) <= self.res:
            reward += 10*MAX_STEPS
            
        if self.steps > MAX_STEPS:
            reward -= MAX_STEPS/5.0
            
            
        # radio reward
        dbm = self.get_dbm_from_point(self.p)
        if self.radio_reward == 1:
            if dbm > self.radio_best_signal_range:
                reward += 5
            elif dbm < self.radio_best_signal_range and dbm > self.average_radio_signal:
                reward += 2
            elif dbm < self.average_radio_signal:
                reward -= 5
        elif self.radio_reward == 2:
            if dbm > self.radio_best_signal_range:
                reward += 5
            elif dbm < self.average_radio_signal:
                reward -= 5
        elif self.radio_reward == 3:
            if dbm < self.average_radio_signal:
                reward -= 20
        elif self.radio_reward == 4:
            norm_dbm = self.normalize_value(self.get_rgb_from_rssi(dbm), 0, 1, -5, 5) 
            reward += norm_dbm
        elif self.radio_reward == 5:
            if dbm > self.average_radio_signal:
                reward += 5
        return reward
    
    ########################################
    # terminou?
    def terminal(self):
        # colisao
        if self.collision(self.p):
            return True
        # chegou no alvo
        if np.linalg.norm(self.p - self.alvo) <= self.res:
            return True
        if self.steps > MAX_STEPS:
            return True
        return False
        
    ########################################
    # ambientes em 2D
    def init2D(self, image):

        # le a imagem
        I = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        # linhas e colunas da imagem
        self.nrow = I.shape[0]
        self.ncol = I.shape[1]

        # binariza imagem
        (thresh, I) = cv2.threshold(I, 127, 255, cv2.THRESH_BINARY)

        # inverte a imagem em y
        self.mapa = cv2.flip(I, 0)

        # parametros de conversao
        self.mx = float(self.ncol) / float(self.xlim[1] - self.xlim[0])
        self.my = float(self.nrow) / float(self.ylim[1] - self.ylim[0])
        
        

    ########################################
    # pega ponto aleatorio no voronoi
    def getRand(self):
        # pega um ponto aleatorio
        while True:
            qx = np.random.uniform(self.xlim[0], self.xlim[1])
            qy = np.random.uniform(self.ylim[0], self.ylim[1])
            q = (qx, qy)
            # verifica colisao
            if not self.collision(q):
                break

        # retorna
        return q

    ########################################
    # verifica colisao com os obstaculos
    def collision(self, q):

        # posicao de colisao na imagem
        px, py = self.mts2px(q)
        col = int(px)
        lin = int(py)

        # verifica se esta dentro do ambiente
        if (lin <= 0) or (lin >= self.nrow):
            return True
        if (col <= 0) or (col >= self.ncol):
            return True

        # colisao
        try:
            if self.mapa.item(lin, col) < 127:
                return True
        except IndexError:
            None

        return False

    ########################################
    # transforma pontos no mundo real para pixels na imagem
    def mts2px(self, q):
        qx, qy = q
        # conversao
        px = (qx - self.xlim[0])*self.mx
        py = self.nrow - (qy - self.ylim[0])*self.my

        return px, py

    ##########################################
    # converte estados continuos em discretos
    def obs_to_state(self, num_states, lower_bounds, upper_bounds, obs):
        state_idx = []
        for ob, lower, upper, num in zip(obs, lower_bounds, upper_bounds, num_states):
            state_idx.append(self.discretize_val(ob, lower, upper, num))

        return np.ravel_multi_index(state_idx, num_states)

    ##########################################
    # discretiza um valor
    def discretize_val(self, val, min_val, max_val, num_states):
        state = int(num_states * (val - min_val) / (max_val - min_val))
        if state >= num_states:
            state = num_states - 1
        if state < 0:
            state = 0
        return state

    ########################################
    # desenha a imagem distorcida em metros
    def render(self, Q):
        
        # desenha o robo
        plt.plot(self.p[0], self.p[1], 'rs')

        # desenha o alvo
        plt.plot(self.alvo[0], self.alvo[1], 'r', marker='x', markersize=20, linewidth=10)

        # plota mapa real e o mapa obsevado
        plt.imshow(self.mapa, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]], alpha=0.5)

        # vector field
        m = self.num_states[0]
        xm = np.linspace(self.xlim[0], self.xlim[1], m)
        ym = np.linspace(self.ylim[0], self.ylim[1], m)
        XX, YY = np.meshgrid(xm, ym)

        th = np.linspace(0.0, 2.0*np.pi, NACTIONS)[:-1]
        vx = []
        vy = []
        for x in xm:
            for y in ym:
                S = self.get_state(np.array([y, x]))
                # plota a melhor ação                
                u = self.actionU(Q[S, :].argmax())
                vx.append(u[0])
                vy.append(u[1])
                    
        Vx = np.array(vx)
        Vy = np.array(vy)
        M = np.hypot(Vx, Vy)
        plt.gca().quiver(XX, YY, Vx, Vy, M, cmap='crest', angles='xy', scale_units='xy', scale=1.5, headwidth=5)
        scatter = plt.scatter(self.x, self.y, c=self.rgb, s=100, marker='s')
        plt.xticks([], fontsize=FONTSIZE)
        plt.yticks([], fontsize=FONTSIZE)
        plt.xlim(self.xlim + 0.05*np.abs(np.diff(self.xlim))*np.array([-1., 1.]))
        plt.ylim(self.ylim + 0.05*np.abs(np.diff(self.ylim))*np.array([-1., 1.]))
        plt.box(True)
        # plt.show()
        plt.pause(.1)

    ########################################
    def log_distance_equation(self, position, antena):
        d = self.calculate_distance_between_two_points(position, antena)
        if d < self.res:
            d = self.res
        PL = self.L0 + 10 * self.n * math.log((d)/self.d0)
        if PL < 0:
            PL = 0
        dbm = self.PT + self.Ar + self.At - PL
        if dbm > self.limit_max_dBm:
            dbm = self.limit_max_dBm
        return dbm
    
    ########################################
    def get_dbm_from_point(self, p):
        antena = self.get_closest_antenna(p[0],p[1])
        dbm = self.log_distance_equation(p, antena)
        return dbm           
    
    ########################################
    def radio_beheaviour(self):
        for x in np.arange(self.xlim[0], self.xlim[1]+self.res, self.res):
            for y in np.arange(self.xlim[0], self.ylim[1]+self.res, self.res):
                antena = self.get_closest_antenna(x,y)
                dbm = self.log_distance_equation([x,y], antena)
                self.x.append(x)
                self.y.append(y)
                self.dbm.append(dbm)
                intensity = self.get_rgb_from_rssi(dbm)
                self.rgb.append([
                    1.0 * intensity,
                    0.8 * (1 - 2.0 * abs(intensity - 1 / 2)),
                    1 - 1.0 * intensity,
                    0.6
                ])
        
        for i in range(len(self.rgb)):
            if self.rgb[i][0] > 1 or self.rgb[i][0] < 0:
                print("red", self.x[i], self.y[i])
            if self.rgb[i][1] > 1 or self.rgb[i][1] < 0:
                print("gree",self.dbm[i])
            if self.rgb[i][2] > 1 or self.rgb[i][2] < 0:
                print("blue", self.dbm[i])
            
            
        self.plot_radio_map()
        
    #########################################
    def plot_radio_map(self):
        plt.imshow(self.mapa, cmap='gray', extent=[self.xlim[0], self.xlim[1], self.ylim[0], self.ylim[1]], alpha=0.8)
        # for x in range (self.xlim[0], self.xlim[1], self.res):
        #     for y in range(self.xlim[0], self.ylim[1], self.res):
        scatter = plt.scatter(self.x, self.y, c=self.rgb, s=100, marker='s')
        plt.title('Mapa da potência do sinal de rádio')
        plt.xlabel('X')
        plt.ylabel('Y')

        print("plotando resultado")

        # Show the plot
        plt.show()
        
                  
    ##########################################
    def get_rgb_from_rssi(self, rssi):
        rgb = ((((rssi+100) - self.radio_min_dBm) * self.NewRange) / self.OldRange_dBm) + self.green_min 
        return rgb
    
    ##########################################
    def get_range(self):
        self.OldRange_dBm = self.radio_max_dBm - self.radio_min_dBm
        self.NewRange = self.green_max - self.green_min

    ##########################################
    def get_closest_antenna(self, x, y):
        old_distance = math.inf
        closest_antenna = []
        for i in range(len(self.antenas)):
            distance = self.calculate_distance_between_two_points(self.antenas[i], (x, y))
            if distance < old_distance:
                closest_antenna = i
                old_distance = distance
        return(self.antenas[closest_antenna])
            
    
    ########################################
    def calculate_distance_between_two_points(self, position1, position2):
        return (math.sqrt((position1[0] - position2[0])**2 + (position1[1] - position2[1])**2))
    
    
    def normalize_value(self, original_value, original_min, original_max, new_min, new_max):
    # Calculate the normalized value using the linear transformation formula
        normalized_value = new_min + ((new_max - new_min) * (original_value - original_min) / (original_max - original_min))
        return normalized_value
    
    
    ########################################
    def __del__(self):
        None
