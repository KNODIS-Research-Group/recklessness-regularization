package es.upm.etsisi.knodis.recklessness;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.MAE;
import es.upm.etsisi.knodis.recklessness.recommender.PMF;
import es.upm.etsisi.knodis.recklessness.recommender.MLP;

public class BaselinesTestSplitError {

    private static String DATASET = "ml100k";

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
        }

        // PMF model

        PMF pmf = null;

        if (DATASET.equals("ml100k")) {
            pmf = new PMF(datamodel, 6, 100, 0.1, 0.01, 4815162341L);
        } else if (DATASET.equals("ml1m")) {
            pmf = new PMF(datamodel, 10, 100, 0.1, 0.01, 4815162341L);
        } else if (DATASET.equals("ft")) {
            pmf = new PMF(datamodel, 2, 100, 0.1, 0.01, 4815162341L);
        }

        pmf.fit();

        double pmfMaeScore = 1 - new MAE(pmf, 0).getScore() / (datamodel.getMaxRating() - datamodel.getMinRating());
        System.out.println("MAE score of PMF recommender=" + pmfMaeScore);

        // MLP model

        MLP mlp = null;

        if (DATASET.equals("ml100k")) {
            mlp = new MLP(datamodel, 2, 100, 0.1, 4815162341L);
        } else if (DATASET.equals("ml1m")) {
            mlp = new MLP(datamodel, 2, 100, 0.1, 4815162341L);
        } else if (DATASET.equals("ft")) {
            mlp = new MLP(datamodel, 8, 100, 0.01, 4815162341L);
        }

        mlp.fit();

        double mlpMaeScore = 1 - new MAE(mlp, 0).getScore() / (datamodel.getMaxRating() - datamodel.getMinRating());
        System.out.println("MAE score of MLP recommender=" + mlpMaeScore);
    }
}
