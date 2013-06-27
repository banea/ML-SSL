import java.util.Comparator;

/*Override of compare function for the PriorityQueue Comparator
 * returns the minimum value (which is stored in the root
 * maintains the highest values
 * */

public class MinimumValueComparator implements Comparator<InstanceValuePair> {
  @Override
  public int compare(InstanceValuePair x, InstanceValuePair y) {
    if (x.getValue() < y.getValue()) {
      return -1;
    }
    if (x.getValue() > y.getValue()) {
      return 1;
    }
    return 0;
  }
}
