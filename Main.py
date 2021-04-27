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
import pygame

def main():
    print("Hello World!")

if __name__ == "__main__":
    main()







def evolve(settings, organisms_old, gen):

    #elitism_num = int(floor(settings['elitism'] * settings['pop_size']))
    new_orgs = settings['pop_size']# - elitism_num

    #--- GET STATS FROM CURRENT GENERATION ----------------+
    stats = defaultdict(int)
    for org in organisms_old:
        if org.fitness > stats['BEST'] or stats['BEST'] == 0:
            stats['BEST'] = org.fitness

        if org.fitness < stats['WORST'] or stats['WORST'] == 0:
            stats['WORST'] = org.fitness
            
        stats['SUM'] += org.fitness
        stats['COUNT'] += 1

    stats['AVG'] = stats['SUM'] / stats['COUNT']
    
    
    #--- ELITISM (KEEP BEST PERFORMING ORGANISMS) ---------+
    orgs_sorted = sorted(organisms_old, key=operator.attrgetter('fitness'), reverse=True)
    organisms_new = []
    #for i in range(0, elitism_num):
    #    organisms_new.append(organism(settings, wih=orgs_sorted[i].wih, who=orgs_sorted[i].who, name=orgs_sorted[i].name))

    
    #--- GENERATE NEW ORGANISMS ---------------------------+
    for w in range(0, new_orgs):

        # SELECTION (TRUNCATION SELECTION)
        canidates = range(0, elitism_num)
        random_index = sample(canidates, 2)
        org_1 = orgs_sorted[random_index[0]]
        org_2 = orgs_sorted[random_index[1]]

        # CROSSOVER
        crossover_weight = random()
        wih_new = (crossover_weight * org_1.wih) + ((1 - crossover_weight) * org_2.wih)
        who_new = (crossover_weight * org_1.who) + ((1 - crossover_weight) * org_2.who)
        
        # MUTATION
        mutate = random()
        if mutate <= settings['mutate']:

            # PICK WHICH WEIGHT MATRIX TO MUTATE
            mat_pick = randint(0,1)     

            # MUTATE: WIH WEIGHTS
            if mat_pick == 0:
                index_row = randint(0,settings['hnodes']-1)
                wih_new[index_row] = wih_new[index_row] * uniform(0.9, 1.1)
                if wih_new[index_row] >  1: wih_new[index_row] = 1
                if wih_new[index_row] < -1: wih_new[index_row] = -1
                
            # MUTATE: WHO WEIGHTS
            if mat_pick == 1:
                index_row = randint(0,settings['onodes']-1)
                index_col = randint(0,settings['hnodes']-1)
                who_new[index_row][index_col] = who_new[index_row][index_col] * uniform(0.9, 1.1)
                if who_new[index_row][index_col] >  1: who_new[index_row][index_col] = 1
                if who_new[index_row][index_col] < -1: who_new[index_row][index_col] = -1
                    
        organisms_new.append(organism(settings, wih=wih_new, who=who_new, name='gen['+str(gen)+']-org['+str(w)+']'))
                
    return organisms_new, stats

def setup():
    #setup organisms
    Settings = settings()
    #setup terrain
    terrain = Terrain()

def simulate(settings, organisms, foods, gen):

    total_time_steps = int(settings['gen_time'] / settings['dt'])
    
    #--- CYCLE THROUGH EACH TIME STEP ---------------------+
    for t_step in range(0, total_time_steps, 1):

        # PLOT SIMULATION FRAME
        #if gen == settings['gens'] - 1 and settings['plot']==True:
        if gen==49:
            plot_frame(settings, organisms, foods, gen, t_step)
        

        # UPDATE FITNESS FUNCTION
        for food in foods:
            for org in organisms:
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # UPDATE FITNESS FUNCTION
                if food_org_dist <= 0.075:
                    org.fitness += food.energy
                    food.respawn(settings)

                # RESET DISTANCE AND HEADING TO NEAREST FOOD SOURCE
                org.d_food = 100
                org.r_food = 0

        # CALCULATE HEADING TO NEAREST FOOD SOURCE
        for food in foods:
            for org in organisms:
                
                # CALCULATE DISTANCE TO SELECTED FOOD PARTICLE
                food_org_dist = dist(org.x, org.y, food.x, food.y)

                # DETERMINE IF THIS IS THE CLOSEST FOOD PARTICLE
                if food_org_dist < org.d_food:
                    org.d_food = food_org_dist
                    org.r_food = calc_heading(org, food)

        # GET ORGANISM RESPONSE
        for org in organisms:
            org.think()

        # UPDATE ORGANISMS POSITION AND VELOCITY
        for org in organisms:
            org.update_r(settings)
            org.update_vel(settings)
            org.update_pos(settings)

    return organisms

class settings():
    def __init__(self):
        #map settings
        self.terrainSize
        #entity settings
        self.visionRange
        #generation settings
        self.generationPopulationSize
        self.generationTime
        self.generationCount
        self.mutationRate
        #neural network settings
        self.hiddenLayerLength = 10


class TerrainNode():
    def __init__(self,moveCost,foodValue,fertilityValue,populationLimit,hazardLevel,sprite):
        self.moveCost = moveCost
        self.foodValue = foodValue
        self.fertilityValue = fertilityValue
        self.populationLimit = populationLimit
        self.population
        self.hazardLevel = hazardLevel
        self.sprite = sprite

class Terrain():
    plains = TerrainNode(1,1,1,10,0,"none")
    water = TerrainNode(1,1,1,10,0,"none")
    forest = TerrainNode(1,1,1,10,0,"none")
    mountain = TerrainNode(1,1,1,10,0,"none")
    nodes = 0
    def __init__(self, sizeX, sizeY):
        self.sizeX = sizeX
        self.sizeY = sizeY
        self.nodes = [sizeX][sizeY]
        self.nodes = self.generateNodes()

    def generateNodes(self):
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.nodes[x][y] = self.plainsplains
        return self.nodes

class Entity():
    def __init__(self):
        self.energyLevel
        self.visionRange
        self.visibleTerrain
        self.logic
        self.position
        self.orientation
        self.name

    def think(self):

        # SIMPLE MLP
        activationFunc = lambda x: np.tanh(x)               # activation function
        layerHidden1 = activationFunc(np.dot(self.weightsInputToHidden, self.inputValues))  # hidden layer
        layerOutput = activationFunc(np.dot(self.weightsHiddenToOutput, layerHidden1))      # output layer

        # UPDATE dv AND dr WITH MLP RESPONSE
        highestActionValue = np.max(layerOutput)
        for index in range(layerOutput):
            if(layerOutput[index] == highestActionValue):
                if(index == 0):
                    self.eat()
                elif(index == 1):
                    self.move()
                elif(index == 2):
                    self.rotateLeft()
                elif(index == 3):
                    self.rotateRight()

    def eat(currentNode):
        if(currentNode.foodValue>=1):
            currentNode.foodValue = currentNode.foodValue - 1
            self.energyLevel = self.energyLevel + 1
        
    def move(self, currentRotation):
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
        if self.position[0] >= Settings.terrainSize[0]: self.position[0] = Settings.terrainSize[0]-1
        if self.position[1] < 0: self.position[1] = 0
        if self.position[0] <= Settings.terrainSize[1]: self.position[1] = Settings.terrainSize[1]-1

    def rotateLeft(self):
        self.currentRotation = self.currentRotation - 1
        if self.currentRotation < 0: self.currentRotation = 7
        if self.currentRotation > 7: self.currentRotation = 0

    def rotateRight(self):
        self.currentRotation = self.currentRotation + 1
        if self.currentRotation < 0: self.currentRotation = 7
        if self.currentRotation > 7: self.currentRotation = 0


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

