package HomeWork4;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

public class Knn implements Classifier {
	
	public enum EditMode {None, Forwards, Backwards};
	private EditMode m_editMode = EditMode.None;
	private Instances m_trainingInstances;


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
		//TODO: implement this method
	}

	private void editedForward(Instances instances) {
		//TODO: implement this method
	}

	private void editedBackward(Instances instances) {
		//TODO: implement this method
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
		for (Instance i: i_data) {
			double iClassified = classifyInstance(i);
			double iTrueClass = i.value(i.classIndex());
			if(iClassified != iTrueClass){
				mistakeSum++;
			}
		}
		double avg = mistakeSum / (double)i_data.numInstances();
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

    public double crossValidationError(Instances i_data){
	    
    }

}
