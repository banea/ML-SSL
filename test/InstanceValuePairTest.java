import static org.junit.Assert.*;
import org.junit.Test;
import weka.core.Instance;

public class InstanceValuePairTest {
  Instance myInstance1 = new Instance(3);
  Instance myInstance2 = new Instance(3);
  double score = 0.748956;
  double label = 0.0;
  InstanceValuePair monolingual = new InstanceValuePair(myInstance1, score,
      label);
  InstanceValuePair bilingual = new InstanceValuePair(myInstance1, myInstance2,
      score, label);

  @Test
  public void testGetLabel() {
    assertTrue("Label", label == monolingual.getLabel());
    assertTrue("Label", label == bilingual.getLabel());
  }

  @Test
  public void testGetInstance() {
    assertTrue("Instance", myInstance1.equals(monolingual.getInstance()));
  }

  @Test
  public void testGetLang1Instance() {
    assertTrue("Instance1", myInstance1.equals(bilingual.getLang1Instance()));
  }

  @Test
  public void testGetLang2Instance() {
    assertTrue("Instance2", myInstance2.equals(bilingual.getLang2Instance()));
  }

  @SuppressWarnings("deprecation")
  @Test
  public void testGetValue() {
    assertTrue("Score monolingual", score == monolingual.getValue());
    assertTrue("Score bilingual", score == bilingual.getValue());
  }

  @Test
  public void testToString() {
    System.out.println(monolingual.toString());
  }

}
