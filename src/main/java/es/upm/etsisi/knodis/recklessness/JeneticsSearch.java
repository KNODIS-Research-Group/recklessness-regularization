package es.upm.etsisi.knodis.recklessness;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.recommender.Recommender;
import es.upm.etsisi.cf4j.util.optimization.GridSearchCV;
import es.upm.etsisi.cf4j.util.optimization.ParamsGrid;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.CummulativeCoverage;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.CummulativeMAE;
import es.upm.etsisi.knodis.recklessness.recommender.BeMF;

import io.jenetics.*;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.Limits;
import io.jenetics.ext.moea.MOEA;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.UFTournamentSelector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.util.ISeq;
import io.jenetics.util.IntRange;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;

import java.io.*;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class JeneticsSearch {

    private static String DATASET = "ml100k";
    private static boolean WITH_RECKLESSESS = false;
    private static long SEED = 4815162342L;
    private static DataModel datamodel = null;
    private static double[] scores = null;
    private static PrintWriter outputCSV;
    private static PrintWriter outputPareto;
    private static final int GENS = 150;
    private static final int POP = 100;

    public static void main(String[] args) throws Exception {

        Random rand = new Random(SEED);
        RandomRegistry.random(rand);

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

        File resultsPath = new File("results/");
        if (!resultsPath.exists()) {
            resultsPath.mkdirs();
        }

        // Create directory for CSV results
        try {
            File outputFile = new File(resultsPath + "/" + DATASET + "-jenetic-search-recklessness-" + ((WITH_RECKLESSESS) ? "yes" : "no") + ".csv");
            outputCSV = new PrintWriter(outputFile);
            printHeader();
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Create directory for Pareto results
        try {
            File outputFile = new File(resultsPath + "/" + DATASET + "-pareto-front-recklessness-" + ((WITH_RECKLESSESS) ? "yes" : "no") + ".txt");
            outputPareto = new PrintWriter(outputFile);
        } catch (IOException e) {
            e.printStackTrace();
        }

        ExecutorService executor = Executors.newSingleThreadExecutor();

        final Engine<IntegerGene, Vec<double[]>> engine =
                Engine.builder(JeneticsSearch::fitness, codec)
                        .executor(executor)
                        .maximizing()
                        .offspringSelector(UFTournamentSelector.ofVec())
                        .alterers(
                                new Crossover(Alterer.DEFAULT_ALTER_PROBABILITY) {
                                    @Override
                                    protected int crossover(MSeq that, MSeq other) {
                                        if(that.get(0).getClass().equals("io.jenetics.IntegerGene")){
                                            IntegerGene firstGene = (IntegerGene) that.get(0);
                                            IntegerGene secondGene = (IntegerGene) other.get(0);
                                            Integer mean = Math.round((firstGene.intValue()+secondGene.intValue())/2);
                                            that.set(0, mean);
                                            other.set(0,mean);
                                        }else if(that.get(0).getClass().equals("io.jenetics.DoubleGene")){
                                            DoubleGene firstGene = (DoubleGene) that.get(0);
                                            DoubleGene secondGene = (DoubleGene) other.get(0);
                                            Double mean = (firstGene.doubleValue()+secondGene.doubleValue())/2;
                                            that.set(0, mean);
                                            other.set(0,mean);
                                        }
                                        return 1;
                                    }
                                },
                                new Mutator<>()
                        )
                        .survivorsSelector(NSGA2Selector.ofVec())
                        .populationSize(POP)
                        .build();

        final ISeq<Phenotype<IntegerGene, Vec<double[]>>> paretoSet =
                engine.stream()
                        .limit(Limits.byFixedGeneration(GENS))
                        .peek(JeneticsSearch::toFile)
                        .collect(MOEA.toParetoSet(IntRange.of(20, 50)));

        System.out.println("\nFinished\n" + paretoSet);

        outputPareto.println(paretoSet);
        outputPareto.close();
        outputCSV.close();
        executor.shutdown();
    }

    private static final Genotype codec = getGenotype();

    private static Genotype getGenotype(){
        Genotype codec = (WITH_RECKLESSESS)?
                            Genotype.of(
                                    (Chromosome) IntegerChromosome.of(0, 300),
                                    (Chromosome) IntegerChromosome.of(0, 200),
                                    DoubleChromosome.of(0.0001, 1.0),
                                    DoubleChromosome.of(0.001, 0.05),
                                    DoubleChromosome.of(-1.0, 1.0))
                            :
                            Genotype.of(
                                    (Chromosome) IntegerChromosome.of(0, 300),
                                    (Chromosome) IntegerChromosome.of(0, 200),
                                    DoubleChromosome.of(0.0001, 1.0),
                                    DoubleChromosome.of(0.001, 0.05));
        return codec;
    }

    private static synchronized Vec<double[]> fitness(final Genotype gt) {

        final IntegerChromosome numItersC = (IntegerChromosome) gt.get(0);
        final IntegerChromosome numFactorsC = (IntegerChromosome) gt.get(1);
        final DoubleChromosome regularizationC = (DoubleChromosome) gt.get(2);
        final DoubleChromosome learningRateC = (DoubleChromosome) gt.get(3);

        DoubleChromosome recklessnessC = null;
        if(WITH_RECKLESSESS)
            recklessnessC = (DoubleChromosome) gt.get(4);

        final int numIters = numItersC.intValue();
        final int numFactors = numFactorsC.intValue();
        final double regularization = regularizationC.doubleValue();
        final double learningRate = learningRateC.doubleValue();

        double recklessness;
        if(WITH_RECKLESSESS)
            recklessness = recklessnessC.doubleValue();
        else
            recklessness = 0.0;

        ParamsGrid paramsGrid = new ParamsGrid();

        paramsGrid.addFixedParam("numIters", numIters);
        paramsGrid.addFixedParam("numFactors", numFactors);
        paramsGrid.addFixedParam("regularization", regularization);
        paramsGrid.addFixedParam("learningRate", learningRate);
        paramsGrid.addFixedParam("recklessness", recklessness);

        paramsGrid.addFixedParam("scores", scores);
        paramsGrid.addFixedParam("seed", SEED);

        double cumulativeMAE = 0.0;
        double cumulativeCoverage = 0.0;

        try {
            GridSearchCV gridSearch = new GridSearchCV(datamodel, paramsGrid, BeMF.class, new Class[]{CummulativeMAE.class, CummulativeCoverage.class}, 5, SEED);
            gridSearch.fit();
            cumulativeMAE = gridSearch.getBestScore(0, false);
            cumulativeCoverage = gridSearch.getBestScore(1, false);
        } catch (Exception e) {
        }

        System.out.println("CumulativeMAE: "+cumulativeMAE+" - CumulativeCOverage: "+cumulativeCoverage);
        return Vec.of(cumulativeMAE, cumulativeCoverage);
    }

    private static void toFile(final EvolutionResult<IntegerGene, Vec<double[]>> result) {
        System.out.println("Exporting generation: " + result.generation());
        String out = "";

        for (int i = 0; i < result.population().length(); i++) {
            Genotype gt = result.population().get(i).genotype();

            out += result.generation() + ";";
            out += i + ";";
            out += gt.get(0).toString() + ";";
            out += gt.get(1).toString() + ";";
            out += gt.get(2).toString() + ";";
            out += gt.get(3).toString() + ";";
            if(WITH_RECKLESSESS)
                out += gt.get(4).toString() + ";";
            out += result.population().get(i).fitness().data()[0] + ";";
            out += result.population().get(i).fitness().data()[1];

            if (i < result.population().length() - 1) {
                out += "\n";
            }
        }

        outputCSV.println(out.replace("[", "").replace("]", ""));
        outputCSV.flush();
    }

    public static void printHeader() {
        String out = "generation;";
        out += "individualId;";
        out += "numIters;";
        out += "numFactors;";
        out += "regularization;";
        out += "learningRate;";
        if(WITH_RECKLESSESS)
            out += "recklessness;";
        out += "cumulativeMAE;";
        out += "cumulativeCoverage";

        outputCSV.println(out);
        outputCSV.flush();
    }
}
