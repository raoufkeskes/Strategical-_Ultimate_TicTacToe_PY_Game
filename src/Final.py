# -*- coding: utf-8 -*-

# Nicolas, 2015-11-18

from __future__ import absolute_import, print_function, unicode_literals
from gameclass import Game,check_init_game_done
from spritebuilder import SpriteBuilder
from players import Player
from sprite import MovingSprite
from ontology import Ontology
from itertools import chain
import pygame
import glo

import random 
import numpy as np
import sys


import heapq
############### PriorityQueue Class pour gérer la frontière ########################

class PriorityQueue:
    def __init__(self):
        self.elements = []
    
    def empty(self):
        return len(self.elements) == 0
    
    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self):
        return heapq.heappop(self.elements)[1]

#####################################################################################

############## GameGrid Class pour représenter la matrice du Jeu ####################
class GameGrid:
    def __init__(self, width, height,walls):
        self.width = width
        self.height = height
        self.walls = walls
        self.weights = {}
    
    #Si un point id=(x,y)  est une case dans le terrain du jeu
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    #Si une Case id=(x,y)  n'est pas une Case Mur
    def passable(self, id):
        return id not in self.walls

    #Retourne tous les voisins valides d'une case id
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x, y-1), (x-1, y), (x, y+1)]
        if (x + y) % 2 == 0: results.reverse() # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results

    #Retourne le chemin optimal d'une case 'start' vers une case 'goal'
    #Elle sera appelée directement après L'execution de l algorithme A*
    def reconstruct_path(self,came_from, start, goal):
        current = goal
        path = []
        while current != start:
            path.append(current)
            current = came_from[current]
        path.append(start) # optional
        path.reverse() # optional
        return path

    #Retourne le coût d'un noeud source vers Noeud Destinataire
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)
    
  
##################################################################################

############################## A Star ALgorithm ##################################

#Distance de manhattan 
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

#Implementation de l'algorithme A*
#Elle retourne :
	#une liste des came_from ( le Noeud Parent Optimal )  de toutes les cases exploitées
	#une liste des coûts de toutes les cases ( Les noeuds ) exploitées
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(goal, next)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

##################################################################################  
# ---- ---- ---- ---- ---- ----
# ---- Main                ----
# ---- ---- ---- ---- ---- ----

game = Game()

def init(_boardname=None):
    global player,game
    # pathfindingWorld_MultiPlayer4
    name = _boardname if _boardname is not None else 'tictactoe'
    game = Game('Cartes/' + name + '.json', SpriteBuilder)
    game.O = Ontology(True, 'SpriteSheet-32x32/tiny_spritesheet_ontology.csv')
    game.populate_sprite_names(game.O)
    game.fps = 2  # frames per second
    game.mainiteration()
    game.mask.allow_overlaping_players = True
    #player = game.player
    
def main():

    #for arg in sys.argv:
    iterations = 200 # default
    if len(sys.argv) == 2:
        iterations = int(sys.argv[1])
    print ("Iterations: ")
    print (iterations)

    init()
    
    
    

    
    #-------------------------------
    # Initialisation
    #-------------------------------
       
    players = [o for o in game.layers['joueur']]
    nbPlayers = len(players)
    #score = [0]*nbPlayers
    fioles = {} # dictionnaire (x,y)->couleur pour les fioles
    
    
    # on localise tous les états initiaux (loc du joueur)
    initStates = [o.get_rowcol() for o in game.layers['joueur']]
    print ("Init states:", initStates)
    
    
    # on localise tous les objets ramassables
    #goalStates = [o.get_rowcol() for o in game.layers['ramassable']]
    #print ("Goal states:", goalStates)
        
    # on localise tous les murs
    wallStates = [w.get_rowcol() for w in game.layers['obstacle']]
    tictactoeStates = [(x,y) for x in range(3,16) for y in range(3,16)]
    #print ("Wall states:", wallStates)
    
    #Préparer Notre Grille de jeu
    g = GameGrid(20, 20,wallStates)
    g.walls = wallStates 
    
    
        
    #-------------------------------
    # Placement aleatoire des fioles de couleur 
    #-------------------------------
    
    for o in game.layers['ramassable']: # on considère chaque fiole
        
        #on détermine la couleur
    
        if o.tileid == (19,0): # tileid donne la coordonnee dans la fiche de sprites
            couleur = 'r'
        elif o.tileid == (19,1):
            couleur = 'j'
        else:
            couleur = 'b'

        # et on met la fiole qqpart au hasard

        x = random.randint(1,19)
        y = random.randint(1,19)

        while (x,y) in tictactoeStates or (x,y) in wallStates or (x,y) in fioles: # ... mais pas sur un mur
            x = random.randint(1,19)
            y = random.randint(1,19)
        o.set_rowcol(x,y)
        # on ajoute cette fiole 
        fioles[(x,y)]=couleur

        game.layers['ramassable'].add(o)
        game.mainiteration()                

    print("Les ", len(fioles), " fioles ont été placées aux endroits suivants: \n", fioles)


    
    
    #-------------------------------
    # Boucle principale de déplacements 
    #-------------------------------
    
        
    # bon ici on fait juste plusieurs random walker pour exemple...
    
    posPlayers = initStates

    for i in range(iterations):
        fioles_couples = [*fioles] ;
        for j in range(nbPlayers): # on fait bouger chaque joueur séquentiellement
            start = posPlayers[j]
            goal  =  fioles_couples[j]
            came_from, cost_so_far = a_star_search(g, start, goal)
            #Calcul du chemin complet de ce joueur
            path=g.reconstruct_path(came_from, start  , goal  )
	    #Parcour de ce chemin optimal dans le jeu
            for couple in path : 
                (next_row,next_col) = couple[0],couple[1]
                players[j].set_rowcol(next_row,next_col)
                game.mainiteration()  

            print ("Objet de couleur ", fioles[goal], " trouvé par le joueur ", j)
	    ## TO DO FOR NEXT ##

	    ## Aller poser la fiole dans la grille
            ## générer une nouvelle fiole 
           
            #END
            
    
    print (fioles)
    pygame.quit()

main()
