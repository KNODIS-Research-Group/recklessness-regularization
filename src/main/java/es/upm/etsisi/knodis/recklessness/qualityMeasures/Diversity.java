package es.upm.etsisi.knodis.recklessness.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.recommender.knn.itemSimilarityMetric.Cosine;
import es.upm.etsisi.cf4j.recommender.knn.itemSimilarityMetric.ItemSimilarityMetric;
import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.knodis.recklessness.recommender.ProbabilistcRecommender;

public class Diversity extends QualityMeasure  {

    private final int numberOfRecommendations;
    private final ProbabilistcRecommender recommender;
    ItemSimilarityMetric itemSimilarityMetric;
    private final double reliabilityThreshold;

    public Diversity(ProbabilistcRecommender recommender, int numberOfRecommendations, double reliabilityThreshold) {
        super(recommender);
        this.recommender = recommender;
        this.numberOfRecommendations = numberOfRecommendations;
        this.reliabilityThreshold = reliabilityThreshold;

        this.itemSimilarityMetric = new Cosine();
        this.itemSimilarityMetric.setDatamodel(super.recommender.getDataModel());
        Parallelizer.exec(recommender.getDataModel().getItems(), this.itemSimilarityMetric);
    }

    @Override
    protected double getScore(TestUser testUser, double[] predictions) {

        int userIndex = testUser.getTestUserIndex();

        // fill with NaN unreliable predictions
        double[] modes = new double[predictions.length];
        double[] means = new double[predictions.length];
        for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
            int testItemIndex = testUser.getTestItemAt(pos);
            TestItem testItem = recommender.getDataModel().getTestItem(testItemIndex);
            int itemIndex = testItem.getTestItemIndex();

            double prob = recommender.predictProba(userIndex, itemIndex);
            modes[pos] = (prob >= reliabilityThreshold) ? predictions[pos] : Double.NaN;
            means[pos] = (prob >= reliabilityThreshold) ? recommender.mean(userIndex, itemIndex) : Double.NaN;
        }

        // find the top reliable predictions to be the recommended items
        int[] recommendations = findTopN(modes, means, this.numberOfRecommendations);

        // compute mean average precision
        double sum = 0;
        int count = 0;

        for (int iPos : recommendations) {
            if (iPos == -1) break;

            int iTestItemIndex = testUser.getTestItemAt(iPos);
            int iItemIndex = super.recommender.getDataModel().getTestItem(iTestItemIndex).getItemIndex();

            double[] similarities = this.itemSimilarityMetric.getSimilarities(iItemIndex);

            for (int jPos : recommendations) {
                if (jPos == -1) break;

                if (iPos != jPos) {
                    int jTestItemIndex = testUser.getTestItemAt(jPos);
                    int jItemIndex = super.recommender.getDataModel().getTestItem(jTestItemIndex).getItemIndex();

                    double sim = similarities[jItemIndex];

                    // Ignore items without common ratings (sim == Double.NEGATIVE_INFINITY)
                    if (!Double.isInfinite(sim)) {
                        sum += sim;
                        count++;
                    }
                }
            }
        }

        return (count == 0) ? Double.NaN : (sum / count);
    }

    public static int[] findTopN(double[] values1, double[] values2, int n) {

        int[] indexes = new int[n];

        double[] aux = new double[n];

        for (int i = 0; i < n; i++) {

            // Search highest value
            double value = Double.NEGATIVE_INFINITY;
            int index = -1;
            for (int v = 0; v < values1.length; v++) {
                if (!Double.isNaN(values1[v])) {
                    if ((values1[v] > value) || (values1[v] == value && (index == -1 || values2[v] > values2[index]))) {
                        value = values1[v];
                        index = v;
                    }
                }
            }

            // If there is no value, fill with -1
            if (index == -1) {
                for (; i < indexes.length; i++) indexes[i] = -1;
            }

            // If there is value, add to solution and continue
            else {
                aux[i] = values1[index];
                values1[index] = Double.NaN;
                indexes[i] = index;
            }
        }

        // Restore modified values
        for (int i = 0; i < n; i++) {
            if (indexes[i] == -1) break;
            values1[indexes[i]] = aux[i];
        }

        return indexes;
    }
}