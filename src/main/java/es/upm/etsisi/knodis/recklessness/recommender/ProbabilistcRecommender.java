package es.upm.etsisi.knodis.recklessness.recommender;

import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.recommender.Recommender;

public abstract class ProbabilistcRecommender extends Recommender {

    public ProbabilistcRecommender (DataModel datamodel) {
        super(datamodel);
    }

    public abstract double predictProba(int userIndex, int itemIndex);

    public abstract double mean(int userIndex, int itemIndex);
}
