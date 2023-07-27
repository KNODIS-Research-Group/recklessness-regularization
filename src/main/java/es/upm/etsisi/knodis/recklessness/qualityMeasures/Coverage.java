package es.upm.etsisi.knodis.recklessness.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.knodis.recklessness.recommender.ProbabilistcRecommender;

public class Coverage extends QualityMeasure {

    private ProbabilistcRecommender recommender;
    private double threshold;

    public Coverage(ProbabilistcRecommender recommender, double threshold) {
        super(recommender);
        this.recommender = recommender;
        this.threshold = threshold;
    }

    @Override
    public double getScore(TestUser testUser, double[] predictions) {
        return Double.NaN; // disabled for this project
    }

    @Override
    public double getScore() {
        int count = 0;

        for (int testUserIndex = 0; testUserIndex < recommender.getDataModel().getNumberOfTestUsers(); testUserIndex++) {
            TestUser testUser = recommender.getDataModel().getTestUser(testUserIndex);
            int userIndex = testUser.getUserIndex();

            for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
                int testItemIndex = testUser.getTestItemAt(pos);
                TestItem testItem = recommender.getDataModel().getTestItem(testItemIndex);
                int itemIndex = testItem.getItemIndex();

                double prob = recommender.predictProba(userIndex, itemIndex);
                double prediction = recommender.predictProba(userIndex, itemIndex);

                if (Double.isNaN(prob) || Double.isNaN(prediction)) { // model has not fitted properly
                    return Double.NaN;
                } else {
                    if (prob >= threshold) {
                        count++;
                    }
                }

            }
        }

        double coverage = (double) count / recommender.getDataModel().getNumberOfTestRatings();
        return coverage;
    }
}