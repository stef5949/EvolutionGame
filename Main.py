"""
TerrainNode
int Movement cost
int food value
int fertility value
int populationLimit
entity[] population
int hazardLevel
img sprite

"""

"""
TerrainNode[][] Map
"""
import numpy as np
import pygame as pg
from collections import defaultdict
from random import sample, random, randint, uniform




clock = pg.time.Clock()
class settings():
    def __init__(self):
        #map settings
        self.terrainSize = 30
        #entity settings
        self.visionRange = 3
        #generation settings
        self.generationPopulationSize = 50
        self.generationTime = 50
        self.generationCount = 100
        self.topPerformerAmount = 10
        self.mutationRate = 0.75
        #neural network settings
        self.hiddenLayerLength = 10
        #pygame settings
        self.FPS = 4
        self.screenWidth = 800
        self.screenHeight = 400

#ToDo rewrite to be compatible with our code
def evolve(settings, organismsOld, generation):

    #elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    organismsNew = settings.popSize# - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organismsOld:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness
            
        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']
    
    
    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    #orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    #organisms_new = []
    #for i in range(0, elitism_num):
    #    organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))

    
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for i in range(0, organismsNew):

        # SELECTION (TRUNCATION SELECTION)
        candidateArray = range(0, settings.topPerformerAmount)
        randomIndices = sample(candidateArray, 2)
        org_1 = organismsOld[randomIndices[0]]
        org_2 = organismsOld[randomIndices[1]]

        # CROSSOVER
        crossoverWeight = random()
        weightsInputToHiddenNew = (crossoverWeight * org_1.weightsInputToHidden) + ((1 - crossoverWeight) * org_2.weightsInputToHidden)
        weightsHiddenToOutputNew = (crossoverWeight * org_1.weightsHiddenToOutput) + ((1 - crossoverWeight) * org_2.weightsHiddenToOutput)
        
        # MUTATION
        mutate = random()
        if mutate <= settings.mutationRate:

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)     

            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings.hiddenLayerLength-1)
                weightsInputToHiddenNew[index_row] = weightsInputToHiddenNew[index_row] * uniform(0.9, 1.1)
                if weightsInputToHiddenNew[index_row] >  1: weightsInputToHiddenNew[index_row] = 1
                if weightsInputToHiddenNew[index_row] < -1: weightsInputToHiddenNew[index_row] = -1
                
            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                weightsHiddenToOutputNew[index_row][index_col] = weightsHiddenToOutputNew[index_row][index_col] * uniform(0.9, 1.1)
                if weightsHiddenToOutputNew[index_row][index_col] >  1: weightsHiddenToOutputNew[index_row][index_col] = 1
                if weightsHiddenToOutputNew[index_row][index_col] < -1: weightsHiddenToOutputNew[index_row][index_col] = -1
                    
        organismsNew.append(Entity(settings = settings,weightsInputToHidden = weightsInputToHiddenNew, weightsHiddenToOutput = weightsHiddenToOutputNew, name='gen['+str(generation)+']-org['+str(i)+']'))
                
    return organismsNew, stats


def setup():
    #setup organisms
    Settings = settings()
    # Setup Pygame
    pg.init()
    screen = pg.display.set_mode((Settings.screenWidth, Settings.screenHeight))
    #setup terrain
    terrain = Terrain(sizeX = Settings.terrainSize, sizeY = Settings.terrainSize)
    for x in range(len(terrain.nodes)):
        for y in range(len(terrain.nodes[x])):
            print(str(terrain.nodes[x][y].foodValue), end = ' ')
        print()

    entityList = []
    for i in range(Settings.generationPopulationSize):
        weightsItoH = []
        weightsHtoO = []
        for hiddenLayerCount in range(Settings.hiddenLayerLength):
            tempWeightsItoH = []
            #food value
            for foodValueCount in range(Settings.visionRange):
                for visionWidthCount in range(3):
                    tempWeightsItoH = tempWeightsItoH + [random()]
            #move cost
            for moveCostCount in range(Settings.visionRange):
                for visionWidthCount in range(3):
                    tempWeightsItoH = tempWeightsItoH + [random()]
            #energy level
            tempWeightsItoH = tempWeightsItoH + [random()]
            weightsItoH = weightsItoH + [tempWeightsItoH]
        
        for outputLayerCount in range(4):
            tempWeightsHtoO = []
            for hiddenLayerCount in range(Settings.hiddenLayerLength):
                tempWeightsHtoO = tempWeightsHtoO + [random()]
            weightsHtoO = weightsHtoO + [tempWeightsHtoO]
        entityList = entityList + [Entity(settings = Settings,weightsInputToHidden=weightsItoH,weightsHiddenToOutput=weightsHtoO,name='gen['+str(0)+']-org['+str(i)+']')]
        entityList[i].randomizePosition()


def simulate(entities, terrain):
    """

        # UPDATE FITNESS FUNCTION
        get food value for all entities.
    """
    # GET ORGANISM RESPONSE
    for entity in entities:
        entity.think()

    #Update terrain
    terrain.regenFood

#ToDo adjust terrain
class TerrainNode():
    def __init__(self,moveCost,foodValue,smallFoodfertilityValue, bigFoodFertilityValue, populationLimit,sprite):
        self.moveCost = moveCost
        self.foodValue = foodValue
        self.smallFoodfertilityValue = smallFoodfertilityValue
        self.bigFoodFertilityValue = bigFoodFertilityValue
        self.populationLimit = populationLimit
        self.population = 0
        self.sprite = sprite

class Terrain():
    plains = TerrainNode(1,3,1,5,1,"none")
    water = TerrainNode(5,2,1,3,1,"none")
    forest = TerrainNode(2,5,3,10,1,"none")
    mountain = TerrainNode(10,1,1,2,1,"none")
    nodes = 0
    terrainOptions = {plains,water,forest,mountain}
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.nodes = sizeX * [sizeY * [0]]
        self.nodes = self.generateNodes()

    def generateNodes(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.nodes[x][y] = randint(0,3)
        return self.nodes
    def regenFood(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                if self.nodes[x][y].foodValue < 1 and randint(1,2) != 2:
                    fertilityValue = randint(self.nodes[x][y].smallFoodfertilityValue,self.nodes[x][y].bigFoodFertilityValue)
                    self.nodes[x][y].foodValue = randint(1,fertilityValue+1)
class Entity():
    def __init__(self, settings, weightsInputToHidden, weightsHiddenToOutput, positionX=0, positionY=0, name=""):
        self.energyLevel = 10
        self.settings = settings
        self.visionRange = settings.visionRange
        self.visibleTerrain = None
        self.weightsInputToHidden = weightsInputToHidden
        self.weightsHiddenToOutput = weightsInputToHidden
        self.position = [positionX,positionY]
        self.orientation = 0
        self.name = name
    
    def __str__(self):
        returnString = ""
        for i in self.weightsInputToHidden:
            returnString = returnString + str(i) + ' '
        returnString = returnString + '\n'
        for i in self.weightsHiddenToOutput:
            returnString = returnString + str(i) + ' '
        returnString = returnString + '\n'
        for i in self.position:
            returnString = returnString + str(i) + ' '
        returnString = returnString + '\n'
        returnString = returnString + self.name + ' '
        returnString = returnString + '\n'
        return returnString

    def randomizePosition(self):
        self.position = [int(random() * self.settings.terrainSize),int(random() * self.settings.terrainSize)]

    def think(self, inputValues, terrain):

        # SIMPLE MLP
        activationFunc = lambda x: np.tanh(x)               # activation function
        layerHidden1 = activationFunc(np.dot(self.weightsInputToHidden, inputValues))  # hidden layer
        layerOutput = activationFunc(np.dot(self.weightsHiddenToOutput, layerHidden1))      # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        highestActionValue = np.max(layerOutput)
        for index in range(layerOutput):
            if(layerOutput[index] == highestActionValue):
                if(index == 0):
                    self.eat(terrain)
                elif(index == 1):
                    self.move()
                elif(index == 2):
                    self.rotateLeft()
                elif(index == 3):
                    self.rotateRight()

    def eat(self, terrain):
        currentNode = terrain.nodes[self.position[0],self.position[1]]
        if(currentNode.foodValue>=1):
            currentNode.foodValue = currentNode.foodValue - 1
            self.energyLevel = self.energyLevel + 1

    def move(self):
        currentRotation = self.orientation
        if currentRotation == 0:
            self.position[1] = self.position[1] - 1
        elif currentRotation == 1:
            self.position[0] = self.position[0] + 1
            self.position[1] = self.position[1] - 1
        elif currentRotation == 2:
            self.position[0] = self.position[0] + 1
        elif currentRotation == 3:
            self.position[0] = self.position[0] + 1
            self.position[1] = self.position[1] + 1
        elif currentRotation == 4:
            self.position[1] = self.position[1] + 1
        elif currentRotation == 5:
            self.position[0] = self.position[0] - 1
            self.position[1] = self.position[1] + 1
        elif currentRotation == 6:
            self.position[0] = self.position[0] - 1
        elif currentRotation == 7:
            self.position[0] = self.position[0] - 1
            self.position[1] = self.position[1] - 1
        if self.position[0] < 0: self.position[0] = 0
        if self.position[0] >= self.settings.terrainSize[0]: self.position[0] = self.settings.terrainSize[0]-1
        if self.position[1] < 0: self.position[1] = 0
        if self.position[0] <= self.settings.terrainSize[1]: self.position[1] = self.settings.terrainSize[1]-1

    def rotateLeft(self):
        self.currentRotation = self.currentRotation - 2
        if self.currentRotation < 0: self.currentRotation = 7
        if self.currentRotation > 7: self.currentRotation = 0

    def rotateRight(self):
        self.currentRotation = self.currentRotation + 2
        if self.currentRotation < 0: self.currentRotation = 7
        if self.currentRotation > 7: self.currentRotation = 0
#ToDo implement visuals
running = True
def main():
    setup()
    Settings = settings()
    while running:
        clock.tick(Settings.FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        for generationNumber in range(0,Settings.generationCount):
            for generationRound in range(0,Settings.generationTime):
                simulate()
            evolve()
        pg.display.flip()

if __name__ == "__main__":
    main()


"""
"""

"""
Logic

=========
A bunch of Brians code to start out
function reproduce
    if(this.entity.position && mate.position next to each other && objective for both == reproduction)
        new entity = half of parent 1 genes + half of parent 2 genes
        mutate(new entity)
            pick 1-2 variables change by random number in range x-y        
=========

pathing code
function calcPath(visibleTerrain,objective,surroundingTerrain):
    if(!objectiveVisible):
        function search - move randomly until objective Node discovered
    Insert super cool search algorithm/machine learning thing
    return PathObject
function calcPathDifficulty(Path)
    pathDifficulty =  Movement cost + hazardLevel + population
    if(pathDifficulty > toleratedDifficulty(mutatable)):
        find new path
function evaluateNode()
    check for food on tile
    if(foodLevel > requiredFoodLevel(mutatable))
        eat
function eat
    get Node food level
    lower Node food level
    raise entity energy level
function move
    calcPath
    retrieve next node on path
    if(next node on path != populated)
        set position = next node on path
    else:
        evaluateNode
        calcPath
        try new path 
        if path also not viable then wait
function rotate
    retrieve current path
    retrieve next node on path
    align orientation with Node
"""

"""
GameManager
function update
    run through everything and update accordingly


"""

