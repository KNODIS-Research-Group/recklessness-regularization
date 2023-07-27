package es.upm.etsisi.knodis.recklessness;

import es.upm.etsisi.cf4j.data.BenchmarkDataModels;
import es.upm.etsisi.cf4j.data.DataModel;
import es.upm.etsisi.cf4j.util.Maths;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.Coverage;
import es.upm.etsisi.knodis.recklessness.qualityMeasures.MAE;
import es.upm.etsisi.knodis.recklessness.recommender.*;
import org.apache.commons.lang3.StringUtils;

import java.io.*;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.List;

public class JeneticsTestSplitError {

    private static final String DATASET = "ml1m";
    private static boolean[] WITH_RECKLESSNESS = {true, false};

    private static double[] RELIABILITIES = Maths.linespace(0.0, 1.0, 20, false);

    private static long SEED = 4815162342L;

    public static void main (String[] args) throws Exception {

        DataModel datamodel = null;
        double[] scores = null;

        if (DATASET.equals("ml100k")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
        } else if (DATASET.equals("ml1m")) {
            datamodel = BenchmarkDataModels.MovieLens1M();
            scores = new double[]{1.0, 2.0, 3.0, 4.0, 5.0};
        } else if (DATASET.equals("ft")) {
            datamodel = BenchmarkDataModels.FilmTrust();
            scores = new double[]{0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        }

        double maxDiff = scores[scores.length-1] - scores[0];

        String filePath = "results/" + DATASET + "-test-split-error.csv";
        BufferedWriter csvWriter = new BufferedWriter(new FileWriter(filePath, false));

        DecimalFormatSymbols symbols = new DecimalFormatSymbols();
        symbols.setDecimalSeparator('.');
        DecimalFormat decimalFormat = new DecimalFormat("0.###########", symbols);

        String header = "reliability;exec;1-mae;coverage;recklessness";
        csvWriter.append(header + "\n");
        csvWriter.flush();

        for (boolean withRecklessness : WITH_RECKLESSNESS) {

            List<String[]>paretoFront = getParetoFront(withRecklessness);
            int paretoIndex = 0;
            for (String[] paretoRow : paretoFront) {

                int numIters = Integer.parseInt(paretoRow[0]);
                int numFactors = Integer.parseInt(paretoRow[1]);
                double regularization = Double.parseDouble(paretoRow[2]);
                double learningRate = Double.parseDouble(paretoRow[3]);
                double recklessness = Double.parseDouble(paretoRow[4]);

                BeMF bemf = new BeMF(datamodel, numFactors, numIters, learningRate, regularization, recklessness, scores, SEED);
                bemf.fit();

                for (double rel : RELIABILITIES) {

                    double mae = new MAE(bemf, rel).getScore();
                    double coverage = new Coverage(bemf, rel).getScore();

                    String useRecklessness = "no";
                    if (withRecklessness)
                        useRecklessness = "yes";

                    String row = decimalFormat.format(rel) + ";"
                            + paretoIndex + ";"
                            + decimalFormat.format(1 - mae / maxDiff) + ";"
                            + decimalFormat.format(coverage) + ";"
                            + useRecklessness;

                    csvWriter.append(row + "\n");
                    csvWriter.flush();
                }
                paretoIndex++;
            }
        }
        csvWriter.close();
    }

    private static List<String[]> getParetoFront (boolean withRecklessness) throws Exception {
        String fileName = "results/" + DATASET + "-pareto-front-recklessness-" + (withRecklessness ? "yes" : "no") +".txt";
        BufferedReader br = new BufferedReader(new FileReader(fileName));
        String pareto = "";
        String st;
        while ((st = br.readLine()) != null)
            pareto += st;
        pareto = StringUtils.replace(pareto, " -> ", ",");
        pareto = pareto.replace("[", "").replace("]","").replace(" ","");
        String[] paretoArray = pareto.split(",");
        List<String[]> paretoMatrix = new ArrayList<String[]>();

        int cont = 0;
        while(cont < paretoArray.length){
            String[] paretoRow = new String[5];
            for (int i = 0; i < 4; i++) {
                paretoRow[i] = paretoArray[i+cont];
            }
            if(withRecklessness){
                paretoRow[4] = paretoArray[4+cont];
                cont += 7;
            }else{
                paretoRow[4] = "0.0";
                cont += 6;
            }
            paretoMatrix.add(paretoRow);

        }

        for (String[] paretoRow:paretoMatrix
             ) {
            for (int i = 0; i < paretoRow.length; i++) {
                System.out.print(paretoRow[i]+";");
            }
            System.out.println();
        }

        return paretoMatrix;
    }
}
