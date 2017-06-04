package HomeWork4;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;

import HomeWork4.Knn.EditMode;
import weka.core.Instances;

public class MainHW4 {

    public static void printStats(Knn cls){
        System.out.print("Cross validation error with k = " + cls.getK() + ", p = " + cls.getP() +
                ", Majority function = " + cls.getMajority());
    }

	public static BufferedReader readDataFile(String filename) {
		BufferedReader inputReader = null;

		try {
			inputReader = new BufferedReader(new FileReader(filename));
		} catch (FileNotFoundException ex) {
			System.err.println("File not found: " + filename);
		}

		return inputReader;
	}

	public static Instances loadData(String fileName) throws IOException {
		BufferedReader datafile = readDataFile(fileName);
		Instances data = new Instances(datafile);
		data.setClassIndex(data.numAttributes() - 1);
		return data;
	}

	public static void main(String[] args) throws Exception {
        //TODO: complete the Main method
		Instances cancer = loadData("cancer.txt");
		Instances glass = loadData("glass.txt");
		glass.randomize(new Random());
		cancer.randomize(new Random());
		Knn cancerClass = new Knn();
		Knn glassClass = new Knn();
		glassClass.allData = new Instances(glass);
		cancerClass.buildClassifier(cancer);
		glassClass.buildClassifier(glass);

		glassClass.m_hasID = true;
		glassClass.findHyperParam();
        cancerClass.findHyperParam();
        double errGlass = glassClass.crossValidationError(glass, 10);
        double err = cancerClass.crossValidationError(cancer, 10);
		printStats(cancerClass);
        System.out.println(" for cancer data is " + err);
        printStats(glassClass);
        System.out.println(" for glass data is " + errGlass);
        double[] conf = cancerClass.calcConfusion(cancer);
        System.out.println("The average Precision for the cancer dataset is: " + conf[0]);
        System.out.println("The average Recall for the cancer dataset is " + conf[1]);
        double time;
        double startTime;
        int[] foldOptions = {glass.size(), 50, 10, 5, 3};
        EditMode[] editOptions = {EditMode.None, EditMode.Forwards, EditMode.Backwards};
        for (int numOfFOlds:foldOptions) {
            printFold(numOfFOlds);
            for (EditMode edit:editOptions) {
                glassClass.setEditMode(edit);
                glassClass.buildClassifier(glass);
                startTime = System.nanoTime();
                double CurrErr = glassClass.crossValidationError(glassClass.getM_trainingInstances(), numOfFOlds);
                time = System.nanoTime();
                time = time - startTime;
                printPart2(CurrErr, numOfFOlds, edit, time, glassClass.m_trainingInstancesUsed);
            }
        }


	}

    private static void printFold(int numOfFOlds) {
        System.out.println("----------------------------\n" +
                "Results for " + numOfFOlds + " folds:\t\t \n" +
                "----------------------------\t\t\n");
    }

    private static void printPart2(double currErr, int numOfFOlds, EditMode edit, double time, int numOfinstances) {
        double avg = time / numOfFOlds;
        int instUse = (numOfinstances * numOfFOlds);
        System.out.println("Cross validation error of None-Edited knn on glass dataset is " + currErr + "\n" +
                        "and the average elapsed time is " + avg + "\n"
                + "The total elapsed time is: " + time + "\n" +
                "The total number of instances used in the classification phase is: " + numOfinstances);
    }


}
