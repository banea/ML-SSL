import weka.core.Instance;


public class InstanceValuePair {
	    	Instance instance1=null;
	    	Instance instance2=null;
	    	double classifierScore=0;
	    	double label=0;
	    	
	    	public InstanceValuePair (Instance inst, double classifScore, double label){
	    		instance1=inst;
	    		classifierScore=classifScore;
	    		this.label=label;
	    	}
	    	
	    	public InstanceValuePair (Instance instLang1, Instance instLang2, double classifScore, double label){
	    		instance1=instLang1;
	    		instance2=instLang2;
	    		this.label=label;
	    		classifierScore=classifScore;
	    	}
	    	
	    	public double getLabel(){
	    		return this.label;
	    	}
	    	
	    	public Instance getInstance(){
	    		return this.instance1;
	    	}
	    	
	    	public Instance getLang1Instance(){
	    		return this.instance1;
	    	}
	    	
	    	public Instance getLang2Instance(){
	    		return this.instance2;
	    	}
	    	
	    	public double getValue(){
	    		return this.classifierScore;
	    	}
	    	
	    	public String toString(){
	    		return "instance Score "+ this.getValue()+" Label "+this.getLabel()+"; ";
	    	}
}
