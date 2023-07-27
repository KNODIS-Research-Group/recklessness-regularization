package es.upm.etsisi.knodis.recklessness.recommender;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.data.Item;
import es.upm.etsisi.cf4j.data.User;
import es.upm.etsisi.cf4j.util.Maths;
import es.upm.etsisi.cf4j.util.process.Parallelizer;
import es.upm.etsisi.cf4j.util.process.Partible;

import java.util.Arrays;
import java.util.Map;
import java.util.Random;

/**
 * Implements Ortega, F., Lara-Cabrera, R., González-Prieto, Á., &amp; Bobadilla, J. (2021). Providing reliability in
 * recommender systems through Bernoulli matrix factorization. Information Sciences, 553, 110-128.
 */
public class BeMF extends ProbabilistcRecommender {

    /** Number of latent factors */
    private final int numFactors;

    /** Number of iterations */
    private final int numIters;

    /** Learning rate */
    private final double learningRate;

    /** Regularization parameter */
    private final double regularization;

    /* Recklessness parameter */
    private final double recklessness;

    /** Discrete scores values **/
    private final double[] scores;

    /** Users factors **/
    private final double[][][] U;

    /** Items factors **/
    private final double[][][] V;

    /**
     * Model constructor from a Map containing the model's hyper-parameters values. Map object must
     * contains the following keys:
     *
     * <ul>
     *   <li><b>numFactors</b>: int value with the number of latent factors.</li>
     *   <li><b>numIters:</b>: int value with the number of iterations.</li>
     *   <li><b>learningRate</b>: double value with the learning rate hyper-parameter.</li>
     *   <li><b>regularization</b>: double value with the regularization hyper-parameter.</li>
     *   <li><b>recklessness</b>: double value with the recklessness hyper-parameter.</li>
     *   <li><b>scores</b>: discrete scores values.</li>
     *   <li><b><em>seed</em></b> (optional): random seed for random numbers generation. If missing,
     *       random value is used.
     * </ul>
     *
     * @param datamodel DataModel instance
     * @param params Model's hyper-parameters values
     */
    public BeMF(DataModel datamodel, Map<String, Object> params) {
        this(
                datamodel,
                (int) params.get("numFactors"),
                (int) params.get("numIters"),
                (double) params.get("learningRate"),
                (double) params.get("regularization"),
                (double) params.get("recklessness"),
                (double[]) params.get("scores"),
                params.containsKey("seed") ? (long) params.get("seed") : System.currentTimeMillis()
        );
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of latent factors
     * @param numIters Number of iterations
     * @param learningRate Learning rate
     * @param regularization Regularization
     * @param recklessness Recklessness
     * @param scores Discrete scores values
     */
    public BeMF(DataModel datamodel, int numFactors, int numIters, double learningRate, double regularization, double recklessness, double[] scores) {
        this(datamodel, numFactors, numIters, learningRate, regularization, recklessness, scores, System.currentTimeMillis());
    }

    /**
     * Model constructor
     *
     * @param datamodel DataModel instance
     * @param numFactors Number of latent factors
     * @param numIters Number of iterations
     * @param learningRate Learning rate
     * @param regularization Regularization
     * @param recklessness Recklessness
     * @param scores Discrete scores values
     * @param seed Seed for random numbers generation
     */
    public BeMF(DataModel datamodel, int numFactors, int numIters, double learningRate, double regularization, double recklessness, double[] scores, long seed) {
        super(datamodel);

        this.numFactors = numFactors;
        this.numIters = numIters;
        this.learningRate = learningRate;
        this.regularization = regularization;
        this.recklessness = recklessness;
        this.scores = scores;

        Random rand = new Random(seed);

        this.U = new double[datamodel.getNumberOfUsers()][scores.length][numFactors];
        for (int u = 0; u < datamodel.getNumberOfUsers(); u++) {
            for (int s = 0; s < scores.length; s++) {
                for (int k = 0; k < numFactors; k++) {
                    this.U[u][s][k] = rand.nextDouble();
                }
            }
        }

        this.V = new double[datamodel.getNumberOfItems()][scores.length][numFactors];
        for (int i = 0; i < datamodel.getNumberOfItems(); i++) {
            for (int s = 0; s < scores.length; s++) {
                for (int k = 0; k < numFactors; k++) {
                    this.V[i][s][k] = rand.nextDouble();
                }
            }
        }
    }

    /**
     * Get the number of factors of the model
     *
     * @return Number of factors
     */
    public int getNumFactors() {
        return numFactors;
    }

    /**
     * Get the number of iterations
     *
     * @return Number of iterations
     */
    public int getNumIters() {
        return numIters;
    }

    /**
     * Get the learning rate parameter of the model
     *
     * @return Learning rate
     */
    public double getLearningRate() {
        return learningRate;
    }

    /**
     * Get the regularization parameter of the model
     *
     * @return Regularization
     */
    public double getRegularization() {
        return regularization;
    }

    /**
     * Get the discrete scores values
     *
     * @return Discrete scores values
     */
    public double[] getScores() {
        return scores;
    }

    @Override
    public void fit() {
        System.out.println("\nFitting " + this.toString());

        for (int iter = 1; iter <= this.numIters; iter++) {
            Parallelizer.exec(this.datamodel.getUsers(), new UpdateUsersFactors());
            Parallelizer.exec(this.datamodel.getItems(), new UpdateItemsFactors());

            if ((iter % 10) == 0) System.out.print(".");
            if ((iter % 100) == 0) System.out.println(iter + " iterations");
        }
    }

    @Override
    public double predict(int userIndex, int itemIndex) {
        double max = this.getProbability(userIndex, itemIndex, 0);
        int index = 0;

        for (int s = 1; s < this.scores.length; s++) {
            double prob = this.getProbability(userIndex, itemIndex, s);
            if (max < prob) {
                max = prob;
                index = s;
            }
        }

        return this.scores[index];
    }

    /**
     * Compute the probability that an user rates an item with the rating at r position
     *
     * @param userIndex Index of the user in the array of Users of the DataModel instance
     * @param itemIndex Index of the item in the array of Items of the DataModel instance
     * @param s Score position on the discrete scores values array
     * @return Prediction probability
     */
    private double getProbability(int userIndex, int itemIndex, int s) {
        double dot = Maths.logistic(Maths.dotProduct(this.U[userIndex][s], this.V[itemIndex][s]));

        double sum = 0;
        for (int l = 0; l < this.scores.length; l++) {
            sum += Maths.logistic(Maths.dotProduct(this.U[userIndex][l], this.V[itemIndex][l]));
        }

        return dot / sum;
    }

    /**
     * Computes a prediction probability
     *
     * @param userIndex Index of the user in the array of Users of the DataModel instance
     * @param itemIndex Index of the item in the array of Items of the DataModel instance
     * @return Prediction probability
     */
    public double predictProba(int userIndex, int itemIndex) {
        double prediction = this.predict(userIndex, itemIndex);

        int s = 0;
        while (this.scores[s] != prediction) {
            s++;
        }

        return this.getProbability(userIndex, itemIndex, s);
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder("BeMF(")
                .append("numFactors=").append(this.numFactors)
                .append("; ")
                .append("numIters=").append(this.numIters)
                .append("; ")
                .append("learningRate=").append(this.learningRate)
                .append("; ")
                .append("regularization=").append(this.regularization)
                .append("; ")
                .append("recklessness=").append(this.recklessness)
                .append(";")
                .append("scores=").append(Arrays.toString(this.scores))
                .append(")");
        return str.toString();
    }

    /**
     * Auxiliary inner class to parallelize user factors computation
     */
    private class UpdateUsersFactors implements Partible<User> {

        @Override
        public void beforeRun() { }

        @Override
        public void run(User user) {
            int userIndex = user.getUserIndex();

            for (int pos = 0; pos < user.getNumberOfRatings(); pos++) {
                int itemIndex = user.getItemAt(pos);
                double rating = user.getRatingAt(pos);

                double[] sigmoid = new double[scores.length];
                double sumSigmoid = 0;
                double sumSigmoidDiff = 0;
                for (int s = 0; s < scores.length; s++) {
                    double dot = Maths.dotProduct(U[userIndex][s], V[itemIndex][s]);
                    sigmoid[s] = Maths.logistic(dot);
                    sumSigmoid += sigmoid[s];
                    sumSigmoidDiff += sigmoid[s] * (1-sigmoid[s]);
                }

                for (int l = 0; l < scores.length; l++) {
                    double error = (scores[l] == rating) ? (1 - sigmoid[l]) : -sigmoid[l];

                    double meanOfTheSquare = 0;
                    double squareOfTheMean = 0;

                    if (recklessness != 0) {

                        double sumF = 0;
                        double sumPartial = 0;

                        for (int k = 0; k < scores.length; k++) {
                            double fk = sigmoid[k] / sumSigmoid;
                            double slk = (l == k) ? 1 : 0;
                            double partial = (slk * sigmoid[k] * (1 - sigmoid[k]) * sumSigmoid - sigmoid[k] * sumSigmoidDiff) / Math.pow(sumSigmoid, 2);

                            meanOfTheSquare += Math.pow(scores[k], 2) * partial;
                            sumF += scores[k] * fk;
                            sumPartial += scores[k] * partial;
                        }

                        meanOfTheSquare /= scores.length;
                        squareOfTheMean = sumF * sumPartial / Math.pow(scores.length, 2);

                    }

                    for (int f = 0; f < numFactors; f++) {
                        double gradient= V[itemIndex][l][f] * error;
                        gradient -= regularization * U[userIndex][l][f];
                        gradient += recklessness * V[itemIndex][l][f] * (meanOfTheSquare - squareOfTheMean);

                        U[userIndex][l][f] += learningRate * gradient;
                    }
                }
            }
        }

        @Override
        public void afterRun() { }
    }

    /**
     * Auxiliary inner class to parallelize item factors computation
     */
    private class UpdateItemsFactors implements Partible<Item> {

        @Override
        public void beforeRun() { }

        @Override
        public void run(Item item) {
            int itemIndex = item.getItemIndex();

            for (int pos = 0; pos < item.getNumberOfRatings(); pos++) {
                int userIndex = item.getUserAt(pos);
                double rating = item.getRatingAt(pos);

                double[] sigmoid = new double[scores.length];
                double sumSigmoid = 0;
                double sumSigmoidDiff = 0;
                for (int s = 0; s < scores.length; s++) {
                    double dot = Maths.dotProduct(U[userIndex][s], V[itemIndex][s]);
                    sigmoid[s] = Maths.logistic(dot);
                    sumSigmoid += sigmoid[s];
                    sumSigmoidDiff += sigmoid[s] * (1-sigmoid[s]);
                }

                for (int l = 0; l < scores.length; l++) {
                    double error = (scores[l] == rating) ? (1 - sigmoid[l]) : -sigmoid[l];

                    double meanOfTheSquare = 0;
                    double squareOfTheMean = 0;

                    if (recklessness != 0) {

                        double sumF = 0;
                        double sumPartial = 0;

                        for (int k = 0; k < scores.length; k++) {
                            double fk = sigmoid[k] / sumSigmoid;
                            double slk = (l == k) ? 1 : 0;
                            double partial = (slk * sigmoid[k] * (1 - sigmoid[k]) * sumSigmoid - sigmoid[k] * sumSigmoidDiff) / Math.pow(sumSigmoid, 2);

                            meanOfTheSquare += Math.pow(scores[k], 2) * partial;
                            sumF += scores[k] * fk;
                            sumPartial += scores[k] * partial;
                        }

                        meanOfTheSquare /= scores.length;
                        squareOfTheMean = sumF * sumPartial / Math.pow(scores.length, 2);

                    }

                    for (int f = 0; f < numFactors; f++) {
                        double gradient = U[userIndex][l][f] * error;
                        gradient -= regularization * V[itemIndex][l][f];
                        gradient += recklessness * U[userIndex][l][f] * (meanOfTheSquare - squareOfTheMean);

                        V[itemIndex][l][f] += learningRate * gradient;
                    }
                }
            }
        }

        @Override
        public void afterRun() { }
    }
}
