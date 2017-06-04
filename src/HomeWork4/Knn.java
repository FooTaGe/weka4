package HomeWork4;

import sun.security.jca.GetInstance;
import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.InstanceComparator;
import weka.core.Instances;

public class Knn implements Classifier {
    public class myKNN{
        public Instance[] m_knn;
        public double[] m_Distances;

        public myKNN(Instance[] i_instances, double[] i_distances){
            m_knn = i_instances;
            m_Distances = i_distances;
        }
    }
    public enum Majority {uniform, weighted};
    public enum EditMode {None, Forwards, Backwards};
    public Instances allData;
    public int m_fold = 10;
    public int m_trainingInstancesUsed;
    private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;
    private Double m_lp;
    private Majority m_majority;
    private int m_K;
    public boolean m_hasID = false;
    private InstanceComparator m_checker = new InstanceComparator();

    public Instances getM_trainingInstances(){
        return m_trainingInstances;
    }

    public int getK(){
        return m_K;
    }

    public Double getP(){
        return m_lp;
    }

    public Majority getMajority(){
        return m_majority;
    }

    public EditMode getEditMode() {
		return m_editMode;
	}
	public void setEditMode(EditMode editMode) {
		m_editMode = editMode;
	}

	@Override
	public void buildClassifier(Instances arg0) throws Exception {
		switch (m_editMode) {
		case None:
			noEdit(arg0);
			break;
		case Forwards:
			editedForward(arg0);
			break;
		case Backwards:
			editedBackward(arg0);
			break;
		default:
			noEdit(arg0);
			break;
		}
	}

	@Override
	public double classifyInstance(Instance instance) {
        myKNN KNN = findNearestNeighbors(instance, m_trainingInstances);
        if (m_majority == Majority.uniform){
            return getClassVoteResult(KNN.m_knn);
        }
        else {
            return getWeightedClassVoteResult(KNN.m_knn, KNN.m_Distances);
        }
	}

	private void editedForward(Instances instances) {
        //make sure it's not empty
        Instances fwInstances = new Instances(allData, 0, 1);

        for (int i = 1; i < instances.size(); i++) {
            Instance currInstance = instances.get(i);
            if (currInstance.classValue() != classifyInstance(currInstance)) {
                fwInstances.add(currInstance);

            }
        }
        m_trainingInstances = fwInstances;
    }


	private void editedBackward(Instances instances) {
        Instances bwInstances = new Instances(instances);
        Instance tempInstance;

        for (int i = instances.numInstances() - 1; 0 < i; i--) {
            tempInstance = bwInstances.instance(i);
            bwInstances.delete(i);
            if (tempInstance.classValue() != classifyInstance(tempInstance)) {
                bwInstances.add(tempInstance);
            }
        }

        m_trainingInstances = bwInstances;
    }

	private void noEdit(Instances instances) {
		m_trainingInstances = new Instances(instances);
	}

	@Override
	public double[] distributionForInstance(Instance arg0) throws Exception {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public Capabilities getCapabilities() {
		// TODO Auto-generated method stub
		return null;
	}

	public double calcAvgError(Instances i_data){
		int mistakeSum = 0;
		if (i_data.numInstances() == 0){
		    return 0;
        }
		for (Instance i: i_data) {
			double iClassified = classifyInstance(i);
			double iTrueClass = i.value(i.classIndex());
			if(iClassified != iTrueClass){
				mistakeSum++;
			}
		}
		double avg = (double)mistakeSum / (double)i_data.numInstances();
		return avg;
	}

	/****
	 *
	 * @param i_data
	 * @return [precision][recall]
	 */
	public double[] calcConfusion(Instances i_data){
		int sumTP = 0;
		int sumFP = 0;
		int sumFN = 0;
        for (Instance i: i_data) {
            double iClassification = classifyInstance(i);
            double iTrueClass = i.value(i.classIndex());
            if (iClassification == 0){
                // True positive
                if (iTrueClass == 0){
                    sumTP++;
                }else{
                    sumFP++;
                }
            }
            else{
                if(iTrueClass == 0){
                    sumFN++;
                }
            }
        }
        double[] ans = new double[2];
        ans[0] = sumTP / ((double)sumTP + sumFP);
        ans[1] = sumTP / ((double)sumTP + sumFN);
        return ans;
    }

    public double crossValidationError(Instances i_data, int i_numFolds) throws Exception {
	    int trainingInstancesUsed = 0;
	    double sumError = 0;
        int size = i_data.numInstances() / i_numFolds;
        int sizeOfExtra = i_data.numInstances() % i_numFolds;
	    int first;
	    int numOfinst;
        for (int i = 0; i < i_numFolds; i++) {
            // adding extra instances to the last fold
            if (i + 1 == i_numFolds){
                numOfinst = size + sizeOfExtra;
            }
            else {
                numOfinst = size;
            }
            first = i * size;

	        Instances Validation = new Instances(i_data, first, numOfinst);
            Instances Training = new Instances(i_data);
            // remove Fold from training set
            int j = first + numOfinst - 1;
            for (; j >= first; j--) {
                Training.delete(j);
            }
            buildClassifier(Training);
            sumError += calcAvgError(Validation);
            trainingInstancesUsed += m_trainingInstances.numInstances();
        }
        m_trainingInstancesUsed = trainingInstancesUsed;
        m_trainingInstances = i_data;
        return sumError / i_numFolds;
    }

    public double getClassVoteResult(Instance[] i_KNN){
        int[] sumOfClass = new int[m_trainingInstances.numClasses()];
        for (Instance i: i_KNN) {
            sumOfClass[(int)i.classValue()]++;
        }
        int max = 0;
        int maxIndex = 0;
        for (int i = 0; i < sumOfClass.length; i++) {
            if (sumOfClass[i] > max){
                max = sumOfClass[i];
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public double getWeightedClassVoteResult(Instance[] i_data, double[] i_distances){
        double[] sumClass = new double[i_data[0].numClasses()];
        for (int i = 0; i < i_data.length; i++) {
            sumClass[(int)i_data[i].classValue()] += (1.0 / Math.pow(i_distances[i], 2));
        }
        int maxInd = 0;
        double maxValue = 0;
        int count = 0;
        for (double i: sumClass) {
            if(i > maxValue){
                maxInd = count;
                maxValue = i;
            }
            count++;
        }
        return maxInd;
    }

    public double distance(Instance i_a, Instance i_b){
        if(m_lp == Double.POSITIVE_INFINITY){
            return distanceInf(i_a, i_b);

        }
        else{
            return distanceLP(i_a, i_b);
        }
    }

    public myKNN findNearestNeighbors(Instance i_inst , Instances i_data){
        int numberOfNeighbors = m_K <= i_data.numInstances() ? m_K : i_data.numInstances();
        Double[] allDistances = new Double[i_data.numInstances()];
        Instance curr;
        for (int i = 0; i < i_data.numInstances(); i++) {
            curr = i_data.instance(i);
            if (m_checker.compare(i_inst, curr) == 0){
                allDistances[i] = Double.POSITIVE_INFINITY;
                continue;
            }
            allDistances[i] = distance(i_inst, i_data.instance(i));
        }


        double[] retDis = new double[numberOfNeighbors];
        Instance[] retInstances = new Instance[numberOfNeighbors];
        double minVal = Double.POSITIVE_INFINITY;
        int minIndex = 0;
        for (int i = 0; i < numberOfNeighbors; i++) {
            for (int j = 0; j < allDistances.length; j++) {

                if (allDistances[j] < minVal){
                    minVal = allDistances[j];
                    minIndex = j;
                }
            }
            retDis[i] = minVal;
            retInstances[i] = i_data.instance(minIndex);
            allDistances[minIndex] = Double.POSITIVE_INFINITY;
        }
        myKNN retKNN = new myKNN(retInstances, retDis);
        return retKNN;
    }


    private double distanceLP(Instance i_a, Instance i_b){
        int i = 0;
        double abs;
        double sum = 0;
        double result;
        // ignoring first attribute (case of ID attribute)
        if (m_hasID){
            i++;
        }
        for (; i < i_a.numAttributes() - 1; i++){
            ///change this
            abs = Math.abs(i_a.value(i) - i_b.value(i));
            sum += Math.pow(abs, m_lp);
        }
        result = Math.pow(sum, 1.0 / m_lp);
        return result;
    }

    private double distanceInf(Instance i_a, Instance i_b){
        int i = 0;
        double result = 0.0;
        //ignoring first attribute (case of ID attribute)
        if (m_hasID){
            i++;
        }
        for (; i < i_a.numAttributes() - 1; i++){
            double abs = Math.abs((i_a.value(i) - i_b.value(i)));
            result = Math.max(result, abs);
        }
        return result;
    }

    public void findHyperParam() throws Exception {
        double minError = Double.POSITIVE_INFINITY;
        int minK = 1; // 1-20
        Double minLP = Double.POSITIVE_INFINITY; // INF , 1 , 2, 3
        Majority minMaj = Majority.uniform;
        Double[] LPOptions = {Double.POSITIVE_INFINITY, 1.0, 2.0, 3.0};
        Majority[] majOptions = {Majority.uniform, Majority.weighted};

        for (int i = 1; i <= 20; i++) {
            m_K = i;
            for (Double lp: LPOptions) {
                m_lp = lp;

                for (Majority maj : majOptions) {
                    m_majority = maj;
                    double err = crossValidationError(m_trainingInstances, m_fold);
                    if (err < minError) {
                        minK = m_K;
                        minLP = m_lp;
                        minMaj = m_majority;
                        minError = err;
                    }
                }
            }
        }
        m_K = minK;
        m_lp = minLP;
        m_majority = minMaj;

    }



}
