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

"""
Entity
int energyLevel
int visionRange
TerrainNode[][] visibleTerrain
Logic logic
tuple coordinate
int orientation

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