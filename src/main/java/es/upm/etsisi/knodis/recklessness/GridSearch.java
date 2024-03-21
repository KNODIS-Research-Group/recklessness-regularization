package es.upm.etsisi.knodis.recklessness;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.cf4j.util.optimization.RandomSearchCV;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.CummulativeCoverage;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.CummulativeMAE;
import es.upm.etsisi.knodis.recklessness.recommender.*;

public class GridSearch {

    private static String DATASET = "ml100k";

    private static double RANDOM_SEARCH_COVERAGE = 0.75;

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;
        double[] scores = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
            scores = new double[]{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        }

        ParamsGrid paramsGrid = null;
        RandomSearchCV search  = null;

        // PMF

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numIters", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("gamma", new double[]{0.01, 0.02, 0.03, 0.04, 0.05});
        paramsGrid.addParam("lambda", new double[]{0.0001, 0.001, 0.01, 0.1});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, PMF.class, CummulativeMAE.class, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/" + DATASET + "-gridsearch-pmf.csv");

        // MLP

        paramsGrid = new ParamsGrid();

        paramsGrid.addParam("numEpochs", new int[]{25, 50, 75, 100});
        paramsGrid.addParam("numFactors", new int[]{2, 4, 6, 8, 10});
        paramsGrid.addParam("learningRate", new double[]{0.01, 0.1, 1.0});
        paramsGrid.addFixedParam("seed", SEED);

        search = new RandomSearchCV(datamodel, paramsGrid, MLP.class, CummulativeMAE.class, 5, RANDOM_SEARCH_COVERAGE, SEED);
        search.fit();
        search.exportResults("results/" + DATASET + "-gridsearch-mlp.csv");
    }
}
