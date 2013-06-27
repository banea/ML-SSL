//Class that calculates precision, recall, F-measure

public class ClassEvaluation {
  int tp = 0;
  int tn = 0;
  int fp = 0;
  int fn = 0;

  public void addCase(boolean predicted_class, boolean actual_class) {
    if (predicted_class == true && actual_class == true) {
      tp++; // case was positive and was predicted positive
    } else if (predicted_class == true && actual_class == false) {
      fp++; // case was negative but was predicted positive
    } else if (predicted_class == false && actual_class == true) {
      fn++; // case was positive but was predicted negative
    } else if (predicted_class == false && actual_class == false) {
      tn++; // case was negative and was predicted negative
    }
  }

  public int getDatasetSize() {
    return tp + tn + fp + fn;
  }

  public double getPrecision() {
    return ((double) tp / (double) (tp + fp));
  }

  public double getRecall() {
    return ((double) tp / (double) (tp + fn));
  }

  public double getFmeasure() {
    return (2 * this.getPrecision() * this.getRecall() / (this.getPrecision() + this
        .getRecall()));
  }

  public double getAccuracy() {
    return (double) (tp + tn) / (double) this.getDatasetSize();
  }

  public String toString() {
    String result = this.getPrecision() + "\t" + this.getRecall() + "\t"
        + this.getFmeasure() + "\t" + this.getAccuracy() + "\n" + "\ttp=" + tp
        + "\ttn=" + tn + "\tfp=" + fp + "\tfn=" + fn;
    return result;
  }
}
