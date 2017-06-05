from __future__ import division
import json
import operator
import math
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.neural_network import MLPClassifier
import warnings
from collections import OrderedDict




class Recipe(object):

    id = None
    cuisine = None
    ingredients = []

    def __init__(self, id, cuisine, ingredients):
        self.id = id
        self.cuisine = cuisine
        self.ingredients = ingredients

    def __str__(self):
        
        returnString = "id: " + str(self.id) + "\ncuisine: " + self.cuisine + "\ningredients: " \
        + self.ingredients.__str__() 
        return returnString
    
class ExtendedRecipe(Recipe):

    probabilities = {}

    def __init__(self, recipe, probabilities):
        super(ExtendedRecipe, self).__init__(recipe.id, recipe.cuisine, recipe.ingredients)
        self.probabilities = probabilities

    def __str__(self):
        
        returnString = "id: " + str(self.id) + "\ncuisine: " + self.cuisine + "\ningredients: " \
        + self.ingredients.__str__() 
        return returnString

    def toJSON(self):

        infoDict = {}
        infoDict["id"] = self.id
        infoDict["cuisine"] = self.cuisine
        infoDict["ingredient"] = self.ingredients
        infoDict["probabilities"] = self.probabilities

        return OrderedDict(infoDict)

def RecipeDecoder(obj):

    return Recipe(obj["id"], obj["cuisine"], obj["ingredients"])

def makeCuisineList(recipeList):
    cuisineList = []
    for recipe in recipeList:
        if recipe.cuisine not in cuisineList:
            cuisineList.append(recipe.cuisine)

    return cuisineList

def n_containing(ingredient, recipeList):

    return sum(1 for recipe in recipeList if ingredient in recipe.ingredients)

def idf(ingredient, recipeList):

    return math.log(len(recipeList) / (n_containing(ingredient, recipeList)))

def sort(Dict):
    sortedDict = sorted(Dict.items(), key = operator.itemgetter(1))
    return sortedDict

def truncate(ingredientDict, recipeList):

    copy = ingredientDict
    # make idfDict
    idfDict = {}
    for ingredient in copy:
        idfDict[ingredient] = math.log(len(recipeList) / (copy[ingredient]))

    #remove ingredients with 3 or less total occurrences
    copy = {ingredient: copy[ingredient] \
    for ingredient in copy if copy[ingredient] > 5}

    # remove ingredients with idf of 5 or more
    copy = {ingredient: copy[ingredient] \
    for ingredient in copy if idfDict[ingredient] > 3.5}

    return copy

def format_X_Data(recipeList, truncatedIngredientDict):
    mlb = MultiLabelBinarizer()
    # create the list of list of ingredients
    data = []
    for recipe in recipeList:
        data.append([ingredient for ingredient in recipe.ingredients if ingredient in truncatedIngredientDict])
    formattedData = mlb.fit_transform([set(entry) for entry in data])

    return formattedData, list(mlb.classes_)

def format_Y_Data(recipeList):
    data = []

    for recipe in recipeList:
        if recipe.cuisine == "italian":
            data.append(1)
        else:
            data.append(0)

    return data

def format_Y_Data_MultiClass(recipeList,cuisineList):
    data = []

    for recipe in recipeList:
        cuisine = recipe.cuisine
        data.append(cuisineList.index(cuisine))

    return data


def predict_recipe(recipe, features, clf):
    featureList = [0 for i in range(len(features))]
    for ingredient in recipe.ingredients:
        if ingredient in features:
            index = features.index(ingredient)
            featureList[index] = 1
    
    return clf.predict(featureList)

def predict_recipe_prob(recipe, features, clf):
    featureList = [0 for i in range(len(features))]
    for ingredient in recipe.ingredients:
        if ingredient in features:
            index = features.index(ingredient)
            featureList[index] = 1
    
    return clf.predict_proba(featureList)

def main():
    
    # load recipeList
    recipeListTotal = json.load(open("train.json"), object_hook = RecipeDecoder)

    #recipeList = recipeListTotal[len(recipeListTotal)//2:len(recipeListTotal)]
    recipeList = recipeListTotal[1000:6000]
    # make ingredient dict
    ingredientDict = {}
    for recipe in recipeList:
        for ingredient in recipe.ingredients:
            if ingredient in ingredientDict:
                ingredientDict[ingredient] += 1
            else:
                ingredientDict[ingredient] = 1


    cuisineList = makeCuisineList(recipeListTotal)

    # make truncated ingredient dict
    truncatedIngredientDict = truncate(ingredientDict, recipeList)

    formatted_X_Data, features = format_X_Data(recipeList, truncatedIngredientDict)

    formatted_Y_Data = format_Y_Data_MultiClass(recipeList, cuisineList)

    # train on a neural network

    clf = MLPClassifier(solver = "adam", alpha=1e-5, hidden_layer_sizes=(len(features)//2), random_state=1)
    clf.fit(formatted_X_Data, formatted_Y_Data)

    correct = 0
    outputList = []

    #for i in range(0,len(recipeListTotal)//2):
    for i in range(0,1000):
        cuisineIndex = predict_recipe(recipeListTotal[i], features, clf)
        if cuisineList.index(recipeListTotal[i].cuisine) == cuisineIndex:
            correct += 1
        probs = predict_recipe_prob(recipeListTotal[i], features, clf)
        probs = probs[0]
        probDict = {}
        for j in range(len(probs)):
            if probs[j] > .1:
                # print cuisineList[j], probs[j]
                probDict[cuisineList[j]] = '%.2f' % probs[j]
        # print "\n"
        sortedProbDict = OrderedDict(sorted(probDict.items(), key=lambda t: t[1], reverse = True))
        outputList.append(ExtendedRecipe(recipeListTotal[i], sortedProbDict))

    print correct
    
    # make JSON
    with open('data.json','w') as outfile:
        outfile.write(json.dumps([recipe.toJSON() for recipe in outputList],outfile, indent=4))


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    main()


