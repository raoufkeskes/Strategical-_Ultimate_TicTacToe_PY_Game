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

grille_gagneeJ1 = set()
grille_gagneeJ2 = set()

game = Game()

def init(_boardname=None):
    global player,game
    # pathfindingWorld_MultiPlayer4
    name = _boardname if _boardname is not None else 'tictactoeBis'
    game = Game('Cartes/' + name + '.json', SpriteBuilder)
    game.O = Ontology(True, 'SpriteSheet-32x32/tiny_spritesheet_ontology.csv')
    game.populate_sprite_names(game.O)
    game.fps = 20  # frames per second
    game.mainiteration()
    game.mask.allow_overlaping_players = True
    #player = game.player
    
def main():

    #for arg in sys.argv:
    iterations = 500 # default
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
    #fioles = {} # dictionnaire (x,y)->couleur pour les fioles
    
    
    # on localise tous les états initiaux (loc du joueur)
    initStates = [o.get_rowcol() for o in game.layers['joueur']]
    print ("Init states:", initStates)
    
    
    # on localise tous les objets ramassables
    #goalStates = [o.get_rowcol() for o in game.layers['ramassable']]
    #print ("Goal states:", goalStates)
        
    # on localise tous les murs
    wallStates = [w.get_rowcol() for w in game.layers['obstacle']]
    # et la zone de jeu pour le tic-tac-toe
    tictactoeStates = [(x,y) for x in range(3,16) for y in range(3,16)]
    #print ("Wall states:", wallStates)
    
    # les coordonnees des tiles dans la fiche
    tile_fiole_jaune = (19,1)
    tile_fiole_bleue = (20,1)
    

    #Préparer Notre Grille de jeu
    g = GameGrid(20, 20,wallStates)
    g.walls = wallStates 
    
    # listes des objets fioles jaunes et bleues
    
    fiolesJaunes = [f for f in game.layers['ramassable'] if f.tileid==tile_fiole_jaune]
    fiolesBleues = [f for f in game.layers['ramassable'] if f.tileid==tile_fiole_bleue]   
    all_fioles = (fiolesJaunes,fiolesBleues) 
    fiole_a_ramasser = (0,0) # servira à repérer la prochaine fiole à prendre
    
    # renvoie la couleur d'une fiole
    # potentiellement utile
    
    def couleur(o):
        if o.tileid==tile_fiole_jaune:
            return 'j'
        elif o.tileid==tile_fiole_bleue:
            return 'b'
    
    #-------------------------------
    # Initialiser  toutes les informations sur nos 9 grilles  
    # Intuitivement  je vais essayer d'avoir toutes les informations sur une grilles les positions ainsi que si ils sont occupées ou 
    # pas   On fera  ça avec  une Liste de 9 listes et chaque liste représente  9 cases  
    # Une case est définie par un tuple ( row , col , content )   j ai pas fait un dictionnaire  car j'aurai besoin du rang
    # de cette case pour définir la prochaine  grille
    #-------------------------------
    Grilles = list()
    for i in range (1,4) :
        for j in range (1,4) :
            start_row = i*4
            start_col = j*4
            Grille = list()
            for k in range(start_row,start_row+3):
                for l in range(start_col,start_col+3):
                    Grille.append([k,l,''])
            Grilles.append(Grille)

    #-------------------------------
    # Placement aleatoire d'une fioles de couleur 
    #-------------------------------
    
    def put_next_fiole(j,t):
        o = all_fioles[j][t]
    
        # et on met cette fiole qqpart au hasard
    
        x = random.randint(1,19)
        y = random.randint(1,19)
    
        while (x,y) in tictactoeStates or (x,y) in wallStates: # ... mais pas sur un mur
            x = random.randint(1,19)
            y = random.randint(1,19)
        o.set_rowcol(x,y)
        # on ajoute cette fiole dans le dictionnaire
        #fioles[(x,y)]=couleur(o)
    
        game.layers['ramassable'].add(o)
        game.mainiteration()
        return (x,y)
        
    ligne_gagnante = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(6,4,2)]
    

    #retourne l indexe d'une case gagnante 
    def win_index ( Grille , color ) :
        for tuplee in ligne_gagnante :
            count = 0
            winindex = 0
            for i in range(3):
                if ( Grille[tuplee[i]][2] != '' and Grille[tuplee[i]][2] == color ) :
                    count += 1
                else :
                    winindex = i 
            if ( count == 2 ) :
                return winindex

    #Définie si une grille est morte ou pas 
    def dead_grid (index) :
        Grille = Grilles[index]
        for tuplee in ligne_gagnante :
            count1 = 0
            count2 = 0
            for i in range(3):
                if ( Grille[tuplee[i]][2] != '' and Grille[tuplee[i]][2] == 'j' ) :
                    count1 += 1
                if ( Grille[tuplee[i]][2] != ''and Grille[tuplee[i]][2] == 'b' ) :
                    count2 += 1
            if ( count1 == 3 ) :
                    grille_gagneeJ1.add(index)
                    is_Win(1)
                    return True 
            if ( count2 == 3 ) :
                    grille_gagneeJ2.add(index)
                    is_Win(2)
                    return True 
        return False   
    def is_Win(joueur):
        for tuplee in ligne_gagnante :
            counter = 0 
            for el in tuplee :
                if ( joueur == 1 ) : grillesGagnee = grille_gagneeJ1
                else : grillesGagnee = grille_gagneeJ2
                if ( el in grillesGagnee ) : counter += 1
            if ( counter == 3 ) :
                print (" Le joueur ",joueur," a gagné")
                sys.exit ()
                
 
    #-------------------------------
    # Boucle principale de déplacements, un joueur apres l'autre
    #-------------------------------
    
    posPlayers = initStates
    nbrTourMax = len(all_fioles[0])   #Le nombre de tour = nombre de fioles d'un des deux joueur    
    nextGrilleIndex = -1
    for tour in range(nbrTourMax):
        for j in range(nbPlayers): # on fait bouger chaque joueur séquentiellement
            ########################## RAMASSER   LA FIOLE ##########################
            fiole_a_ramasser = put_next_fiole(j,tour)
            start = posPlayers[j]
            goal  =  fiole_a_ramasser
            #Appliquer A*
            came_from, cost_so_far = a_star_search(g, start, goal)
            #Calcul du chemin complet de ce joueur
            path=g.reconstruct_path(came_from, start  , goal  )
            #Parcour de ce chemin optimal dans le jeu
            for couple in path : 
                (next_row,next_col) = couple[0],couple[1]
                players[j].set_rowcol(next_row,next_col)
                game.mainiteration()  
            o = players[j].ramasse(game.layers) # on la ramasse
            game.mainiteration()
            posPlayers[j] = (next_row,next_col)
            ########################## RAMASSER   FIN ###############################


            ########################## DEPOSER  LA FIOLE ##########################
            if ( nextGrilleIndex == -1 ) :
                    nextGrille =  Grilles[random.randint(0,8)]

            else : 
                    nextGrille =  Grilles[nextGrilleIndex]
            
           
            #Joueur 1 stratégie Coins dans le sens de la montre 
            if ( j == 0 ) :
                #Selectionner une case
                coins = [0,2,8,6] 
                coinindex = 0
                #Tant que la case est remplie on cherchera une autre
                while ( coinindex < 4 and nextGrille[coins[coinindex]][2] != '' ):
                    coinindex = coinindex + 1
                    
                if ( coinindex <= 3 ) :
                    caseindex = coins[coinindex]
                else :
                    caseindex = win_index ( nextGrille , couleur(o) )  
                
            #Joueur 2 Aléatoire
            if ( j == 1 ) :
                #Selectionner une case vide aléatoire
                randomCaseIndex = random.randint(0,8)
                #Tant que la case est remplie on cherchera une autre
                while ( nextGrille[randomCaseIndex][2] != '' ):
                    randomCaseIndex = random.randint(0,8)
                caseindex = randomCaseIndex


            tupleCaseCible = ( nextGrille[caseindex][0] , nextGrille[caseindex][1] )
            rang_Case = caseindex
            start = posPlayers[j]
            goal  = tupleCaseCible
            nextGrille[caseindex][2] = couleur(o)
            #Appliquer A*
            came_from, cost_so_far = a_star_search(g, start, goal)
            #Calcul du chemin complet de ce joueur
            path=g.reconstruct_path(came_from, start  , goal  )
            #Parcour de ce chemin optimal dans le jeu
            for couple in path : 
                (next_row,next_col) = couple[0],couple[1]
                players[j].set_rowcol(next_row,next_col)
                game.mainiteration()  
            #Fixer la nouvelle position du joueur
            posPlayers[j] = (next_row,next_col)
            #Poser la fiole dans son endroit
            o = all_fioles[j][tour]
            o.set_rowcol(goal[0],goal[1])
            game.layers['ramassable'].add(o)
            game.mainiteration()

            #Preparer la prochaine grille pour le prochain joueur
            #la prochaine grille a le meme rang ( dans la grande grille )  que le rang  de la case qu on vient de choisir 
            #dans la petite grille
            print ("Next Case : ", rang_Case )
            nextGrilleIndex = rang_Case
            while(  dead_grid( nextGrilleIndex )    ): nextGrilleIndex = random.randint(0,8)
            ########################## DEPOSER  FIN ###############################

    print ("Egalité entre les deux joueurs ")
    pygame.quit()
    
        
    
   

if __name__ == '__main__':
    main()
    


