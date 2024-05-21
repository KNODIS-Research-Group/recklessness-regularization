package es.upm.etsisi.knodis.recklessness;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.Diversity;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.MAE;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.MAP;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.Novelty;
import es.upm.etsisi.knodis.recklessness.recommender.PMF;
import es.upm.etsisi.knodis.recklessness.recommender.MLP;

public class BaselinesTestSplitError {

    private static String DATASET = "ml100k";

    private static int NUMBER_OF_RECOMMENDATIONS = 10;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;
        Double like_threshold = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens100K();
            like_threshold = 4.0;
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            like_threshold = 4.0;
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
            like_threshold = 3.0;
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

        double pmfMapScore = new MAP(pmf, NUMBER_OF_RECOMMENDATIONS, like_threshold, 0).getScore();
        System.out.println("MAP score of PMF recommender=" + pmfMapScore);

        double pmfNoveltyScore = new Novelty(pmf, NUMBER_OF_RECOMMENDATIONS, 0).getScore();
        System.out.println("Novelty score of PMF recommender=" + pmfNoveltyScore);

        double pmfDiversityScore = new Diversity(pmf, NUMBER_OF_RECOMMENDATIONS, 0).getScore();
        System.out.println("Diversity score of PMF recommender=" + pmfDiversityScore);

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

        double mlpMapScore = new MAP(mlp, NUMBER_OF_RECOMMENDATIONS, like_threshold, 0).getScore();
        System.out.println("MAP score of PMF recommender=" + mlpMapScore);

        double mlpNoveltyScore = new Novelty(mlp, NUMBER_OF_RECOMMENDATIONS, 0).getScore();
        System.out.println("Novelty score of PMF recommender=" + mlpNoveltyScore);

        double mlpDiversityScore = new Diversity(mlp, NUMBER_OF_RECOMMENDATIONS, 0).getScore();
        System.out.println("Diversity score of PMF recommender=" + mlpDiversityScore);
    }
}
