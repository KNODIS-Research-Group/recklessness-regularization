package es.upm.etsisi.knodis.recklessness.qualityMeasures;

import es.upm.etsisi.cf4j.data.TestItem;
import es.upm.etsisi.cf4j.qualityMeasure.QualityMeasure;
import es.upm.etsisi.cf4j.data.TestUser;
import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.cf4j.util.process.Partible;
import es.upm.etsisi.knodis.recklessness.recommender.ProbabilistcRecommender;

public class MAE extends QualityMeasure {

    private ProbabilistcRecommender recommender;
    private double threshold;

    private double score;
    private double[] userErrors;
    private int[] userCounts;
    boolean recommenderHasErrors;

    public MAE(ProbabilistcRecommender recommender, double threshold) {
        super(recommender);
        this.recommender = recommender;
        this.threshold = threshold;
    }

    @Override
    public double getScore(TestUser testUser, double[] predictions) {
        return Double.NaN; // disabled for this project
    }

    @Override
    public double getScore(int numThreads) {
        Parallelizer.exec(recommender.getDataModel().getTestUsers(), new EvaluateUsers(), numThreads);
        return score;
    }

    @Override
    public double getScore() {
        return this.getScore(-1);
    }

    private class EvaluateUsers implements Partible<TestUser> {

        @Override
        public void beforeRun() {
            userErrors = new double[recommender.getDataModel().getNumberOfTestUsers()];
            userCounts = new int[recommender.getDataModel().getNumberOfTestUsers()];
        }

        @Override
        public void run(TestUser testUser) {
            int userIndex = testUser.getUserIndex();
            int testUserIndex = testUser.getTestUserIndex();

            for (int pos = 0; pos < testUser.getNumberOfTestRatings(); pos++) {
                int testItemIndex = testUser.getTestItemAt(pos);
                TestItem testItem = recommender.getDataModel().getTestItem(testItemIndex);
                int itemIndex = testItem.getItemIndex();

                double prob = recommender.predictProba(userIndex, itemIndex);
                double prediction = recommender.predict(userIndex, itemIndex);

                if (Double.isNaN(prob) || Double.isNaN(prediction)) { // model has not fitted properly
                    userErrors[testUserIndex] = Double.NaN;
                    break;
                } else {
                    if (prob >= threshold) {
                        double rating = testUser.getTestRatingAt(pos);
                        userErrors[testUserIndex] += Math.abs(rating - prediction);
                        userCounts[testUserIndex]++;
                    }
                }

            }
        }

        @Override
        public void afterRun() {
            double sumErrors = 0;
            double sumCounts = 0;

            for (int i = 0; i < userErrors.length; i++) {
                if (Double.isNaN(userErrors[i])) {
                    score = Double.NaN;
                    break;
                } else {
                    sumErrors += userErrors[i];
                    sumCounts += userCounts[i];
                }
            }

            if (!Double.isNaN(score)) {
                score = sumErrors / sumCounts;
            }
        }
    }
}