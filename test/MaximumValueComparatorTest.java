import static org.junit.Assert.*;

import java.util.Comparator;
import java.util.PriorityQueue;

import org.junit.Test;
import weka.core.Instance;


public class MaximumValueComparatorTest {
	
	Instance myInstance1 = new Instance (3);
	Instance myInstance2 = new Instance (3);
	double score1 = 0.748956;
	double score2 = 0.25436;
	int label = 0;
	InstanceValuePair monolingual1 = new InstanceValuePair(myInstance1, score1, label);
	InstanceValuePair monolingual2 = new InstanceValuePair(myInstance2, score2, label);
	
	MaximumValueComparator myMaxComparator = new MaximumValueComparator();

	@Test
	public void testCompare() {
		System.out.println(myMaxComparator.compare(monolingual1, monolingual2) + " " + score1);
		assertTrue("Max comparison", -1 == myMaxComparator.compare(monolingual1, monolingual2));
	}
	
	@Test
	public void testPriorityQueueComparator(){
		Comparator<InstanceValuePair> maxComparator = new MaximumValueComparator();
		PriorityQueue<InstanceValuePair> bestObj = new PriorityQueue<InstanceValuePair>(5, maxComparator);
		bestObj.add(new InstanceValuePair(myInstance1, 0.235, 0));
		bestObj.add(new InstanceValuePair(myInstance1, 0.134, 0));
		bestObj.add(new InstanceValuePair(myInstance1, 0.399, 0));
		InstanceValuePair result =  new InstanceValuePair(myInstance1, 0.399, 0);
		bestObj.add(new InstanceValuePair(myInstance1, 0.399, 0));
		bestObj.add(result);
		InstanceValuePair removed = bestObj.remove();
		System.out.println(removed.getValue());
		assertTrue("Instance value pairs have identical scores", result.getValue() == removed.getValue());
	}

}
