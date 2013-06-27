import static org.junit.Assert.*;

import java.util.Comparator;
import java.util.PriorityQueue;

import org.junit.Test;

import weka.core.Instance;


public class MinimumValueComparatorTest {
	Instance myInstance1 = new Instance (3);
	Instance myInstance2 = new Instance (3);
	double score1 = 0.748956;
	double score2 = 0.25436;
	int label = 0;
	InstanceValuePair monolingual1 = new InstanceValuePair(myInstance1, score1, label);
	InstanceValuePair monolingual2 = new InstanceValuePair(myInstance2, score2, label);
	
	MinimumValueComparator myMinComparator = new MinimumValueComparator();


	@Test
	public void testCompare() {
		assertTrue("Min comparison", 1 == myMinComparator.compare(monolingual1, monolingual2));
	}
	
	@Test
	public void testPriorityQueueComparator(){
		Comparator<InstanceValuePair> minComparator = new MinimumValueComparator();
		PriorityQueue<InstanceValuePair> bestObj = new PriorityQueue<InstanceValuePair>(5, minComparator);
		InstanceValuePair result =  new InstanceValuePair(myInstance1, 0.134, 0);
		bestObj.add(result);
		bestObj.add(new InstanceValuePair(myInstance1, 0.235, 0));
		bestObj.add(new InstanceValuePair(myInstance1, 0.399, 0));
		
		InstanceValuePair removed = bestObj.remove();
		System.out.println(removed.getValue());
		assertTrue("Instance value pairs have identical scores", result.getValue() == removed.getValue());
	}

}
