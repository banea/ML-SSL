import static org.junit.Assert.*;

import org.junit.Test;


public class ClassEvaluationTest {
	ClassEvaluation myEvaluation = new ClassEvaluation();;

	public void generateSetup(int tn_examples, int tp_examples, int fn_examples, int fp_examples) {
		for (int i = 0; i < tn_examples; i++){
			myEvaluation.addCase(false, false);
		}
		for (int i = 0; i< fp_examples; i++){
			myEvaluation.addCase(true, false);
		}
		for (int i = 0; i< fn_examples; i++){
			myEvaluation.addCase(false, true);
		}
		for (int i = 0; i< tp_examples; i++){
			myEvaluation.addCase(true, true);
		}
	}

	@Test
	public void testGetDatasetSize(){
		this.generateSetup (9760, 60, 40, 140);
		assertEquals("Number of samples in the dataset", 10000, myEvaluation.getDatasetSize());
	}
	
	@Test
	public void testGetPrecision() {
		this.generateSetup (9760, 60, 40, 140);
		System.out.println (myEvaluation.getPrecision());
		assertTrue("Precision", 0.3 == myEvaluation.getPrecision());;
	}

	@Test
	public void testGetRecall() {
		this.generateSetup (9760, 60, 40, 140);
		System.out.println (myEvaluation.getRecall());
		assertTrue("Recall", 0.6 == myEvaluation.getRecall());;
	}

	@Test
	public void testGetFmeasure() {
		this.generateSetup (9760, 60, 40, 140);
		System.out.println (myEvaluation.getFmeasure());
		assertTrue("Fmeasure", 0.4 == myEvaluation.getFmeasure());;
	}

	@Test
	public void testGetAccuracy() {
		this.generateSetup (9760, 60, 40, 140);
		System.out.println (myEvaluation.getAccuracy());
		assertTrue("Accuracy", 0.982 == myEvaluation.getAccuracy());;
	}
}
