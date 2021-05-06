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
from pygame.locals import *
from collections import defaultdict
from random import sample, random, randint, uniform


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
    terrainOptions = [plains,water,forest,mountain]
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.nodes = [[TerrainNode(0,0,0,0,0,"none") for columns in range(sizeY)] for rows in range(sizeX)]
        self.nodes = self.generateNodes()

    def generateNodes(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.nodes[x][y] = self.terrainOptions[randint(0,3)]
        return self.nodes
    def regenFood(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                if self.nodes[x][y].foodValue < 1 and randint(1,2) != 2:
                    fertilityValue = randint(self.nodes[x][y].smallFoodfertilityValue,self.nodes[x][y].bigFoodFertilityValue)
                    self.nodes[x][y].foodValue = randint(1,fertilityValue+1)
class Entity():
    def __init__(self, settings, weightsInputToHidden, weightsHiddenToOutput, positionX=0, positionY=0, name=""):
        self.energyLevel = 15
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

    def setVision(self,terrain):
        self.visibleTerrain = [[TerrainNode(0,0,0,0,0,"none") for columns in range(self.visionRange)] for rows in range(3)]
        #if(self.orientation==0):
        for visionRow in range(len(self.visibleTerrain)):
            for visionColumn in range(len(self.visibleTerrain[visionRow])):
                if(self.orientation==0):
                    terrainRow = self.position[0] - 1 + visionRow
                    terrainColumn = self.position[1] + 1 - self.visionRange + visionColumn
                elif(self.orientation==1):
                    terrainRow = self.position[1] + 1 - self.visionRange + visionColumn
                    terrainColumn = self.position[0] - 1 + visionRow
                elif(self.orientation==2):
                    terrainRow = self.position[0] + 1 - visionRow
                    terrainColumn = self.position[1] - 1 + self.visionRange - visionColumn
                elif(self.orientation==3):
                    terrainRow = self.position[1] - 1 + self.visionRange - visionColumn
                    terrainColumn = self.position[0] + 1 - visionRow

                if (terrainRow < 0 or terrainRow >= terrain.sizeX) or (terrainColumn < 0 or terrainColumn >= terrain.sizeY):
                    self.visibleTerrain[visionRow][visionColumn] = TerrainNode(0,0,0,0,0,"none")
                else:
                    self.visibleTerrain[visionRow][visionColumn] = terrain.nodes[terrainRow][terrainColumn]


    def think(self, terrain):
        if(self.energyLevel>0):
            self.setVision(terrain)
            # SIMPLE MLP
            foodValues = []
            moveCosts = []
            for i in range(len(self.visibleTerrain)):
                for j in range(len(self.visibleTerrain[i])):
                    foodValues += [self.visibleTerrain[i][j].foodValue]
                    moveCosts += [self.visibleTerrain[i][j].moveCost]
            inputValues = foodValues + moveCosts + [self.energyLevel, self.orientation]
            activationFunc = lambda x: np.tanh(x)               # activation function
            layerHidden1 = activationFunc(np.dot(self.weightsInputToHidden, inputValues))  # hidden layer
            #layerOutput = activationFunc(np.dot(self.weightsHiddenToOutput, layerHidden1))      # output layer
            layerOutput = []
            for outputIndex in range(4):
                outputValue = 0
                for hiddenIndex in range(len(layerHidden1)):
                    outputValue+=layerHidden1[hiddenIndex] * self.weightsHiddenToOutput[outputIndex][hiddenIndex]
                layerOutput+=[activationFunc(outputValue)]

            highestActionValue = np.max(layerOutput)
            for index in range(len(layerOutput)):
                if(layerOutput[index] == highestActionValue):
                    if(index == 0):
                        self.eat(terrain)
                        #print(self.name + " is eating")
                        self.energyLevel+=0.005
                    elif(index == 1):
                        self.move()
                        self.energyLevel+=0.01
                        if(terrain.nodes[self.position[0]][self.position[1]].foodValue>0):
                            self.energyLevel+=0.1
                        #print(self.name + " is moving")
                    elif(index == 2):
                        self.rotateLeft()
                        #print(self.name + " is rotating left")
                    elif(index == 3):
                        self.rotateRight()
                        #print(self.name + " is rotating right")
            self.energyLevel-=0.3

    def eat(self, terrain):
        currentNode = terrain.nodes[self.position[0]][self.position[1]]
        if(currentNode.foodValue>=1):
            currentNode.foodValue = currentNode.foodValue - 1
            self.energyLevel += 1

    def move(self):
        currentRotation = self.orientation
        if currentRotation == 0:
            self.position[1] = self.position[1] - 1
        elif currentRotation == 1:
            self.position[0] = self.position[0] + 1
        elif currentRotation == 2:
            self.position[1] = self.position[1] + 1
        elif currentRotation == 3:
            self.position[0] = self.position[0] - 1
        if self.position[0] < 0: self.position[0] = 0
        if self.position[0] >= self.settings.terrainSize: self.position[0] = self.settings.terrainSize-1
        if self.position[1] < 0: self.position[1] = 0
        if self.position[0] <= self.settings.terrainSize: self.position[1] = self.settings.terrainSize-1

    def rotateLeft(self):
        self.orientation += 1
        if self.orientation < 0: self.orientation = 3
        if self.orientation > 3: self.orientation = 0

    def rotateRight(self):
        self.orientation -= 1
        if self.orientation < 0: self.orientation = 3
        if self.orientation > 3: self.orientation = 0


clock = pg.time.Clock()
class settings():
    def __init__(self):
        #map settings
        self.terrainSize = 30
        #entity settings
        self.visionRange = 5
        #generation settings
        self.generationPopulationSize = 25
        self.generationTime = 50
        self.generationCount = 100
        self.topPerformerAmount = 5
        self.mutationRate = 0.75
        #neural network settings
        self.hiddenLayerLength = 10
        #pygame settings
        self.FPS = 30
        self.screenWidth = 800
        self.screenHeight = 400

#ToDo rewrite to be compatible with our code
def evolve(settings, organismsOld, generation):

    organismAmount = settings.generationPopulationSize

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organismsOld:
        if org.energyLevel > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.energyLevel

        if org.energyLevel < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.energyLevel
            
        stats['SUM'] += org.energyLevel
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']

    organismsNew = []
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for organismCounter in range(0, organismAmount):

        # SELECTION
        candidateArray = range(0, settings.topPerformerAmount)
        randomIndices = sample(candidateArray, 2)
        org_1 = organismsOld[randomIndices[0]]
        org_2 = organismsOld[randomIndices[1]]

        # CROSSOVER
        crossoverWeight = random()
        weightsInputToHiddenNew = []
        for i in range(len(org_1.weightsInputToHidden)):
            newWeights = []
            for j in range(len(org_1.weightsInputToHidden[i])):
                newWeights += [(crossoverWeight * org_1.weightsInputToHidden[i][j]) + ((1 - crossoverWeight) * org_2.weightsInputToHidden[i][j])]
            weightsInputToHiddenNew.append(newWeights)
        weightsHiddenToOutputNew = []
        for i in range(len(org_1.weightsHiddenToOutput)):
            newWeights = []
            for j in range(len(org_1.weightsHiddenToOutput[i])):
                newWeights += [(crossoverWeight * org_1.weightsHiddenToOutput[i][j]) + ((1 - crossoverWeight) * org_2.weightsHiddenToOutput[i][j])]
            weightsHiddenToOutputNew.append(newWeights)
        
        # MUTATION
        mutate = random()
        if mutate <= settings.mutationRate:

            # ALWAYS MUTATE AN INPUT WEIGHT
            # input amount: three * visionrange times two (for movecost and food level) + energy level and rotation for each hidden layer node
            #inputWeightAmt = (3 * settings.visionRange * 2 + 2) * settings.hiddenLayerLength
            index_row = randint(0,len(weightsHiddenToOutputNew)-1)
            index_column = randint(0,len(weightsHiddenToOutputNew[0])-1)
            weightsInputToHiddenNew[index_row][index_column] = weightsInputToHiddenNew[index_row][index_column] * uniform(0.9, 1.1)
            if weightsInputToHiddenNew[index_row][index_column] >  1: weightsInputToHiddenNew[index_row][index_column] = 1
            if weightsInputToHiddenNew[index_row][index_column] < -1: weightsInputToHiddenNew[index_row][index_column] = -1

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            random_pick = randint(0,1)     

            # MUTATE: WIH WEIGHTS
            if random_pick == 0:
                index_row = randint(0,len(weightsHiddenToOutputNew)-1)
                index_column = randint(0,len(weightsHiddenToOutputNew[0])-1)
                weightsInputToHiddenNew[index_row][index_column] = weightsInputToHiddenNew[index_row][index_column] * uniform(0.9, 1.1)
                if weightsInputToHiddenNew[index_row][index_column] >  1: weightsInputToHiddenNew[index_row][index_column] = 1
                if weightsInputToHiddenNew[index_row][index_column] < -1: weightsInputToHiddenNew[index_row][index_column] = -1
                
            # MUTATE: WHO WEIGHTS
            if random_pick == 1:
                #outputWeightAmt = 4 * settings.hiddenLayerLength
                index_row = randint(0,3)
                index_column = randint(0,settings.hiddenLayerLength-1)
                weightsHiddenToOutputNew[index_row][index_column] = weightsHiddenToOutputNew[index_row][index_column] * uniform(0.9, 1.1)
                if weightsHiddenToOutputNew[index_row][index_column] >  1: weightsHiddenToOutputNew[index_row][index_column] = 1
                if weightsHiddenToOutputNew[index_row][index_column] < -1: weightsHiddenToOutputNew[index_row][index_column] = -1

        organismsNew.append(Entity(settings = settings,weightsInputToHidden = weightsInputToHiddenNew, weightsHiddenToOutput = weightsHiddenToOutputNew, name="gen["+str(generation)+"]-org["+str(organismCounter)+"]"))
                
    return organismsNew, stats


def setup():
    #setup organisms
    Settings = settings()
    # Setup Pygame
    pg.init()
    screen = pg.display.set_mode((Settings.screenWidth, Settings.screenHeight))
    #setup terrain
    terrainObj = Terrain(sizeX = Settings.terrainSize, sizeY = Settings.terrainSize)

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
            #rotation
            tempWeightsItoH += [random()]
            weightsItoH.append(tempWeightsItoH)
        
        
        for outputLayerCount in range(4):
            tempWeightsHtoO = []
            for hiddenLayerCount in range(Settings.hiddenLayerLength):
                tempWeightsHtoO = tempWeightsHtoO + [random()]
            weightsHtoO.append(tempWeightsHtoO)
        entityList = entityList + [Entity(settings = Settings,weightsInputToHidden=weightsItoH,weightsHiddenToOutput=weightsHtoO,name='gen['+str(0)+']-org['+str(i)+']')]
        entityList[i].randomizePosition()
    return entityList, terrainObj, Settings


def simulate(entities, terrain):
    """

        # UPDATE FITNESS FUNCTION
        get food value for all entities.
    """
    # GET ORGANISM RESPONSE
    for entity in entities:
        entity.think(terrain)

    #Update terrain
    terrain.regenFood

#ToDo adjust terrain

#ToDo implement visuals

def main():
    running = True
    EntitiesObj, TerrainObj, SettingsObj = setup()
    while running:
        clock.tick(SettingsObj.FPS)
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        for generationNumber in range(0,SettingsObj.generationCount):
            TerrainObj.regenFood()
            for generationRound in range(0,SettingsObj.generationTime):
                simulate(EntitiesObj, TerrainObj)
            EntitiesObj, Stats = evolve(SettingsObj, EntitiesObj, generationNumber)
            print("Generation: " + str(generationNumber) + "\nFitness Best: "+str(Stats['BEST']) + "\nFitness Worst: "+str(Stats['WORST']) + "\nFitness Average: "+str(Stats['AVG']) + "\n")
        running = False
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

