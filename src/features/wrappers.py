from src.features.optimisers.evolutionary_operators import crossovers, mutations, parent_selections, \
    population_selections
from src.features.optimisers.geneticAlgorithm import GeneticAlgorithm
from src.features.optimisers.geneticAlgorithmInnovative import GeneticAlgorithmInnovative
from src.features.optimisers.geneticAlgorithmNostalgic import GeneticAlgorithmNostalgic
from src.features.optimisers.geneticAlgorithmAM import GeneticAlgorithmAM

global_params = {
    "populationSize": 100,
    "evaluations": 10000,

}

fs_wrappers = [
    # GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                  crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.RouletteWheel(),
    #                  population_selections.MuPlusLambda()),
    # GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                  crossovers.SinglePointCrossover(), mutations.BitFlipMutation(),
    #                  parent_selections.StochasticUniversalSampling(),
    #                  population_selections.MuPlusLambda()),

    GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
                     crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Tournament(k=2),
                     population_selections.MuPlusLambda()),
#
# GeneticAlgorithm(global_params["populationSize"], 30000, 0.9,
#                      crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Tournament(k=2),
#                      population_selections.MuPlusLambda()),
#
# GeneticAlgorithm(global_params["populationSize"], 50000, 0.9,
#                      crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Tournament(k=2),
#                      population_selections.MuPlusLambda()),

    # GeneticAlgorithmNostalgic(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                           crossovers.SinglePointCrossover(), mutations.BitFlipMutation(),
    #                           parent_selections.Tournament(k=2),
    #                           population_selections.MuPlusLambda()),
    #
    # GeneticAlgorithmInnovative(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                            crossovers.SinglePointCrossover(), mutations.BitFlipMutation(),
    #                            parent_selections.Tournament(k=2),
    #                            population_selections.MuPlusLambda()),

    # GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                  crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Tournament(k=2),
    #                  population_selections.Generational()),

    # GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                  crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Rank(),
    #                  population_selections.MuPlusLambda()),
    #
    # GeneticAlgorithm(global_params["populationSize"], global_params["evaluations"], 0.9,
    #                  crossovers.SinglePointCrossover(), mutations.BitFlipMutation(), parent_selections.Random(),
    #                  population_selections.MuPlusLambda()),
    # GeneticAlgorithmAM(global_params["populationSize"], global_params["evaluations"], 0.9, 0.2, crossovers.SinglePointCrossover(), mutations.AdaptiveMutation(), parent_selections.RouletteWheel(), population_selections.MuPlusLambda()),
]
