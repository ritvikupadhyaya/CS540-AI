import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * Fill in the implementation details of the class DecisionTree using this file.
 * Any methods or secondary classes that you want are fine but we will only
 * interact with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */

public class DecisionTreeImpl extends DecisionTree {

	private DecTreeNode root;
	private List<String> labels; // ordered list of class labels
	private List<String> attributes; // ordered list of attributes
	private Map<String, List<String>> attributeValues; // map to ordered
	// discrete values taken
	// by attributes

	//infoGainArray stores all the values from the attributes, i.e Gain(A)
	private List<Double> infoGainArray;

	private Integer parentAttributeValue;

	//This is a blank DecTreeNode which is used in the calculation of accuracy
	//for a particular subtree starting at the treeNode. This is also used to
	//implement the classify method for the particular subtree instead of from
	//the root all the time (Again used in calculation of accuracy).
	private DecTreeNode treeNode;

	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully

	}

	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train: the training set
	 */
	DecisionTreeImpl(DataSet train) {
		//Initialise the required instance variables
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		//Init the infoGainArray
		this.infoGainArray = new ArrayList<Double>();
		//Init the treeNode to hold the value of null because we do not have a 
		//subtree that we want to get the accuracy or classification of
		this.treeNode = null;
		//Build the decision tree and store the root of that tree in the root
		//variable
		this.root = DecisionTreeLearning(train.instances, attributes, 
				train.instances, -1);
	}

	/**
	 * Build a decision tree given a training set then prune it using a tuning
	 * set.
	 * 
	 * @param train: the training set
	 * @param tune: the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		//Initialise the required instance variables
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		//Init the infoGainArray
		this.infoGainArray = new ArrayList<Double>();
		//Build the decision tree and store the root of that tree in the root
		//variable
		this.root = DecisionTreeLearning(train.instances, attributes, 
				train.instances, -1);

		//Init the treeNode to hold the value of null because we do not have a 
		//subtree that we want to get the accuracy or classification of
		this.treeNode = null;
		//ru:We can check using null, do not need to make a new object from the
		//start. 
		//this.treeNode = new DecTreeNode(-1, -1, -10, true);

		//Call for the pruning of the tree
		prune(tune);

	}

	/**
	 * A helper to the actual DecisionTreeLearning, this method just sets the
	 * global parentAttributeValue to match the method description given in 
	 * the book for DecisionTreeLearning.
	 * 
	 * @param examples list of train's examples given to us to build the tree
	 * @param attributes list of train's examples given to us to build the tree
	 * @param parent_examples list of examples of the parents of the node
	 * @param parentAttributeValue the attribute values of the parent, that is
	 * 								needed to implement a new DecTreeNode
	 * @return the root of the decision tree created
	 */
	private DecTreeNode DecisionTreeLearning(List<Instance> examples, 
			List<String> attributes,
			List<Instance> parent_examples, 
			Integer parentAttributeValue) {

		this.parentAttributeValue = parentAttributeValue;

		return DecisionTreeLearning(examples, attributes, parent_examples);

	}

	/**
	 * DecisionTreeLearning implemented from algorithm in the book (fig 18.5).
	 * This method build the tree from the train data set given to us. This is
	 * the default tree we will get.
	 * 
	 * @param examples list of train's examples given to us to build the tree
	 * @param attributes list of train's examples given to us to build the tree
	 * @param parent_examples list of examples of the parents of the node
	 * @return the root of the decision tree created
	 */
	public DecTreeNode DecisionTreeLearning(List<Instance> examples, 
			List<String> attributes,
			List<Instance> parent_examples) {

		//This DecTreeNode variable will be used to create the tree, this will
		//be the root of the tree
		DecTreeNode newDicisionNode;

		//If there are no examples to train on then return a tree with one node
		//which has the PLURALITY-VALUE of parent_examples as a label, -1 as the
		//attribute because this will be arbitrary value. And we will make it
		//a terminal node.
		if(examples.isEmpty()){
			newDicisionNode = new DecTreeNode(pluralityValue(parent_examples),
					-1, parentAttributeValue, true);
			return newDicisionNode;
		}

		//Now, if there are some examples to train on, we have to make sure
		//that all the examples should have the same classification.
		//So do this, we run each example's label against the first example's
		//label as all should be the same. So if any one of them is not the same
		//as the other, we will set sameClassification to false.
		boolean sameClassification = true;

		int exampleLabel = examples.get(0).label;
		for (int i = 0; i < examples.size(); i++){
			if(examples.get(i).label != exampleLabel){
				sameClassification = false;
				break;
			}
		}

		//If all examples do not have the same classification, we make another
		//default node like we did if there was no example.
		if(sameClassification){
			newDicisionNode = new DecTreeNode(exampleLabel, 0,
					parentAttributeValue, true);
			return newDicisionNode;
		}

		//We check to see if there are any attributes given to us or not. If not
		//then we will again make a default tree like before in the case of no
		//examples
		else if(attributes.isEmpty()){
			newDicisionNode = new DecTreeNode(pluralityValue(examples),
					0, parentAttributeValue, true);
			return newDicisionNode;
		}
		//If we have the proper conditions to build a tree, we will get started 
		else{

			//We get the index that the attribute will have in the attributes
			//list for the attribute which we find is the best one for the tree
			int attributeAIndex = importance(attributes, examples);
			//We then get the actual attribute from the attributes list
			String attributeA = attributes.get(attributeAIndex);

			//Then we have to make a new decision tree with node being tested 
			//against the attribute we decided is the most important.

			//But first we see if the node should be a root node or not. If it
			//is then we also have to set out global variable root to this node
			if(parentAttributeValue == -1){
				root = new DecTreeNode(pluralityValue(examples),
						attributeAIndex,parentAttributeValue,false);
				newDicisionNode = root;

			}
			else{
				newDicisionNode = new DecTreeNode(pluralityValue(examples),
						attributeAIndex,parentAttributeValue,false);
			}

			//Then we go over each value of the selected attribute
			for(int i = 0; i < attributeValues.get(attributeA).size(); i++)
			{
				//We then update the examples list to select only the ones
				//that match the value of the attribute.
				List<Instance> exs = new ArrayList<Instance>();
				for(Instance e: examples){
					if(e.attributes.get(attributeAIndex).equals(i))
						exs.add(e);
				}

				//We then get another new subtree over the reduced examples and
				//attributes and then add that subtree to the existing tree that
				//has been build so far
				newDicisionNode.addChild(
						DecisionTreeLearning(exs, attributes, examples,i));
			}
			//We then finally return our tree that we made
			return newDicisionNode;
		}
	}

	/**
	 * Method votes for the most commonly seen label in all the instances and 
	 * returns its index.
	 * @param examples list of instances that we will go over
	 * @return the index of the most popular label in the instances
	 */
	private int pluralityValue(List<Instance> examples){
		//If the examples are empty, then we can not do anything so we return
		//-1
		if (examples.isEmpty())
			return -1;

		//maxVotes will hold the total number of maximum votes any label has
		//received and instanceIndexPV will hold the index of that label
		int maxVotes = 0, instanceIndexPV = 0;

		//This array list is our vote bank to store every label's frequency 
		List<Integer> labelVoteBank  = new ArrayList<Integer>();

		//just so that we do not get index out of bounds error
		for(int i = 0; i < labels.size(); i++)
		{
			labelVoteBank.add(0);
		}

		//Cast each vote by going through all examples one by one
		for(Instance example: examples){
			labelVoteBank.set(example.label, 
					labelVoteBank.get(example.label)+1);
		}

		//loopCounter helps us keep instanceIndexPV up to date
		int loopCounter = 0;
		//Go through the entire vote bank and see which label has the most votes
		for(Integer voteCount: labelVoteBank){
			//On finding the maximum number of votes so far, we update the 
			//values
			if(voteCount > maxVotes){
				instanceIndexPV = loopCounter;
				maxVotes = voteCount;
			}
			loopCounter++;
		}
		//return the index of the most frequently seen label 
		return instanceIndexPV;
	}

	/**
	 * Importance method goes over a list of attributes and finds the most
	 * suited one based on the entropy values it calculates.
	 * @param a the list of attributes to go over
	 * @param examples the list of instances needed to calculate the entropy
	 * @return the index of the attribute that is best suited. 
	 */
	private int importance(List<String> a, List<Instance> examples){

		//Calculate entropy of the examples
		double entropyOfV = calculateGoalEntropy(examples);
		//Then we calculate the entropy of all the attributes given to us.
		double[] remainderOfA = calculateRemainingEntropy(examples, a);

		//Stores the index of the Attribute which is has the highest information
		//gain.
		int maxIGIndex = 0;
		//Stores the value of the highest information gain in the infoGainArray
		double maxIG = 0;

		//We initialise out infoGainArray which will store the values of all
		//the differences in the Entropy(V) and Remainder(A) for the different
		//values of A. This is also called information gain.
		infoGainArray = new ArrayList<Double>();

		//Store the infoGain values for all the values we got in remainder(A)
		for(int i = 0; i < remainderOfA.length; i++){
			infoGainArray.add(entropyOfV - remainderOfA[i]);
			//If the current value of info gain is higher than the max we have
			//seen so far, update the required values
			if(infoGainArray.get(i) > maxIG){
				maxIG = infoGainArray.get(i);
				maxIGIndex = i;
			}
		}

		//Return the index of the highest info gain we saw
		return maxIGIndex;

	}

	/**
	 * Calculates the entropy on the whole instance set
	 * @param examples the list of instances that we have
	 * @return entropy of the goal attribute
	 */
	private double calculateGoalEntropy(List<Instance> examples){

		//The total number of unique labels that are present in the labels list
		int totalLabels = labels.size(); 
		//We will store the number of times we see a label in this array
		List<Integer> labelCountBank = new ArrayList<Integer>();
		//Just to get rid of index out of bounds
		for(int i = 0;i <totalLabels;i++){
			labelCountBank.add(0);
		}

		//We tally the occurrences of the labels and add it to the set places
		for(Instance example: examples)
			labelCountBank.set(example.label, 
					labelCountBank.get(example.label)+1);


		//We then calculate the probability of lables using the formula given
		//in the book
		List<Double> labelProbability = new ArrayList<Double>();

		for(int j = 0; j < totalLabels; j++){
			//Add the respective probability from basic prob formula to the list
			labelProbability.add(((double)labelCountBank.get(j))
					/ examples.size());
		}

		//We then sum up the entropy of individuals and use the formula given
		//entropyOfV will store the sum of the values
		double entropyOfV = 0.0;
		for(int i = 0; i < totalLabels; i++){
			//If the probability is 0, we just add 0 else the log of 0 will
			//cause a crash
			entropyOfV += (labelProbability.get(i) < 0.0)? 0.0 : 
				(labelProbability.get(i)*
						(Math.log(labelProbability.get(i)) / Math.log(2)));
		}
		//The final value is the negative of this sum
		return -entropyOfV;
	}


	/**
	 * This method goes over all the attributes and gives a list of expected
	 * entropy remaining after the testing attribute A
	 * @param examples list of all the examples
	 * @param attributes list of all the attributes
	 * @return the array containing the remaining entropy of all the attributes
	 */
	private double[] calculateRemainingEntropy(List<Instance> examples, 
			List<String> attributes){

		//This will be the array that will be returned
		double[] attriProb = new double[attributes.size()];

		//Using an arrayList is much harder because with array, incrementing 
		//values is a one like step. Access is much easier

		//We again need to get the number of times this attribute comes up
		//So we store it here, this 2-d array will be the size of the total 
		//number of attributes and each row will be as long as the size of
		//each attribute
		double[][] attributeCountBank = new double[attributes.size()][];

		//Creating the 2-D array and appropriating the size
		for(int i = 0; i < attributes.size(); i++){
			int attriSize = attributeValues.get(attributes.get(i)).size();
			attributeCountBank[i] = new double[attriSize];
		}

		//Go over each example in the examples list and increment the count 
		//in the array where the example's attributes are seen.
		for(Instance example: examples){
			for(int i = 0; i < example.attributes.size(); i++){
				attributeCountBank[i][example.attributes.get(i)]++;
			}
		}

		//We will need to go over all the attributes one by one
		for(String attri: attributes){
			//The remaining entropy for each attribute will be stored here
			double currTotalEntropy = 0.0;

			//Store ALL the probabilities for all the attributes and their
			//instances. 
			double[][] attribureProbability = 
					new double[attributeCountBank[attributeIndex(attri)].length]
							[labels.size()];

			//Again going example by example and counting where ever we see
			//an attribute and incrementing its count
			for(Instance example: examples){
				attribureProbability[example.attributes.get
				                     (attributeIndex(attri))][example.label]++;
			}

			//Get the probability by diving by all the number of times its seen
			//by the total number. Same as we did for the other calculateEntropy
			//method
			for(int i = 0; i < attribureProbability.length; i++){
				for(int j = 0; j < labels.size(); j++){
					attribureProbability[i][j] = attribureProbability[i][j] / 
							attributeCountBank[attributeIndex(attri)][i];
				}
			}

			//We now calculate the log part for each attribute's value
			//before we sum over all of them to get the total remaining entropy
			for(int i = 0; i < attributeCountBank[attributeIndex(attri)].length;
					i++){

				double currAttributeEntopy = 0.0;
				//Over all the probabilities
				for(int j = 0; j < attribureProbability[i].length; j++){
					//Calculate the log part, dividing by log of 2 changes the 
					//base to two.
					if(attribureProbability[i][j] > 0){
						currAttributeEntopy += attribureProbability[i][j] * 
								(Math.log(attribureProbability[i][j]) / 
										Math.log(2));
					}
				}
				//Actual Entropy is the minus of itself
				currAttributeEntopy = -currAttributeEntopy; 
				currTotalEntropy += 
						(attributeCountBank[attributeIndex(attri)][i]
								/examples.size()) * currAttributeEntopy;
			}
			//Set the value in the attriProb array to be returned
			attriProb[attributeIndex(attri)] = currTotalEntropy;
		}
		//return the array
		return attriProb;
	}


	/**
	 * At times it became long or difficult to keep track of where we are using
	 * the index of the attribute or the value (string) of the attribute.
	 * To make it easy to get the index of an attribute in the attribute list,
	 * I made this helper method
	 * @param attribute the attribute we want to look up in the attributes list
	 * @return the index of attribute in the attributes list if found else -1
	 * 
	 */
	private int attributeIndex(String attribute){
		for(int i = 0; i < attributes.size(); i++){
			//If the attribute is found, return that index
			if(attributes.get(i).equals(attribute))
				return i;
		}
		return -1;
	}

	/**
	 * Method calls the required prune method after calculating the accuracy of
	 * the original decision tree on the tune
	 */
	private void prune(DataSet tune){
		//From the lecture slides' pruning algorithm we see

		//Step 1: Calculate the accuracy of T on TUNE. A(T).
		double accuracyRoot = treeAccuracy(tune.instances, root);

		//Step 2:Traverse through the entire tree and get the pruning started
		prune(root, tune, accuracyRoot);
	}

	/**
	 * Method traverses through the tree and for each node and compares the 
	 * accuracy for the original tree provided and the subtree. Finally removes
	 * all the non-required nodes from the tree.
	 * 
	 * @param node root of the part of the tree being considered for pruning
	 * @param tune the DataSet that is used to decide which branch to prune
	 * @param rootAccuracy the accuracy of the original tree before pruning
	 * @return the final tree after pruning is completed for the branch <i>node
	 * </i>
	 */
	private DecTreeNode prune(DecTreeNode node, DataSet tune, 
			double rootAccuracy){

		//Check to see if the node has children, if it does, go over all the
		//children of the node
		if(!node.terminal)
			for(DecTreeNode childNode: node.children)
				//Call the same method on the left most child and then move
				//right from there. Recursion will help us traverse the tree 
				//from bottom left to top.
				prune(childNode, tune, rootAccuracy);

		//Step 2a: Set the node as a terminal node i.e remove its subtree
		node.terminal = true;

		//Step 2c:Checking the accutacy of Tn on tune
		double newTreeAccuracy = treeAccuracy(tune.instances, root);

		//Step 3:If the new tree has a better accuracy then prune, else return
		//the original tree passed into the method
		node.terminal = (newTreeAccuracy >= rootAccuracy);

		//Return the resultant tree
		return node;
	}

	/**
	 * treeAccuracy method calculates the accuracy of the given tree over the 
	 * tune data set.
	 * @param examples is the tune dataset's instance list for measuring accuracy
	 * @param node is the tree or part of the tree whose accuracy we want to 
	 * 			   calculate
	 * @return returns the calculated value of the accuracy of the given tree
	 */
	private double treeAccuracy(List<Instance> examples, DecTreeNode node){
		//Set the treeNode to the given root of the tree or subtree so that the 
		//classify method will calculate the classification of this tree or 
		//subtree and not the root of the original subtree
		treeNode = node;
		//Holds the count of the number of times the classification matches
		double matchCount = 0.0;

		//Go over all the instances of the tune and see if the instance matches
		//the classification
		for(Instance example: examples){
			//Holds the index of the label received from the classify() method
			int labelIndex = -1;
			//Gets the classified label from the classification of the instance
			String classifiedLabel = classify(example);
			//Loops over all the labels to see what is the index of the
			//classified label and then if found, updates labelIndex
			for(int i = 0;i < labels.size(); i++){
				if(labels.get(i).equals(classifiedLabel)){
					labelIndex = i;
				}
			}
			//This checks if the instance's label matches the index we got from
			//the classification of the instance.
			//If so, increment the matchCount and continue with the loop
			if(example.label == labelIndex)
				matchCount++;
		}
		//Set the treeNode to null to make sure that classify() again works
		//for the original decision tree till this method is called next time.
		treeNode = null;
		//Return the accuracy which is the average matches over the entire tune
		//dataset.
		return matchCount/examples.size();

	}

	/* 
	 * Implementation of the classify method
	 */
	@Override
	public String classify(Instance instance) {

		//By default it is the classification of the entire original tree over
		//the given instance
		DecTreeNode node = root;
		//But if we need to classify another tree or a subtree, we change the
		//node to reference that subtree or tree
		if(treeNode != null){
			node = treeNode;
		}

		//Stores the label that was classified for the tree over the instance
		String returnLabel = null;

		//While we do not hit the end of the tree
		while(!node.terminal){

			//TODO:????? ANS: Because we access the
			//instance's attributes not the current list of all attributes
			//NOTE:I get the attribute from the node and then get the correct
			//child by matching that attribute from list of attributes' index
			//String nodeAttribute= attributes.get(node.attribute);			
			//node = node.children.get(attributeIndex(nodeAttribute));

			//We get the attribute's index from the instance's list of attribute
			//at the node's attribute
			int nodeAttributeValue = instance.attributes.get(node.attribute);
			//then we get the related child from the list of node's children
			//who match this attribute
			node = node.children.get(nodeAttributeValue);
			//Then we get the label of the child and update the returnLabel
			//NOTE: the node's label gives the value of the label which has to
			//be used to get the actual label from the labels list
			returnLabel = labels.get(node.label);
		}
		//return the final value of the returnLabel
		return returnLabel;

	}



	@Override
	/**
	 * Print the decision tree in the specified format
	 */
	public void print() {

		printTreeNode(root, null, 0);
	}

	/**
	 * Prints the subtree of the node
	 * with each line prefixed by 4 * k spaces.
	 */
	public void printTreeNode(DecTreeNode p, DecTreeNode parent, int k) {
		StringBuilder sb = new StringBuilder();
		for (int i = 0; i < k; i++) {
			sb.append("    ");
		}
		String value;
		if (parent == null) {
			value = "ROOT";
		} else{
			String parentAttribute = attributes.get(parent.attribute);
			value = attributeValues.get(parentAttribute).get(p.parentAttributeValue);
		}
		sb.append(value);
		if (p.terminal) {
			sb.append(" (" + labels.get(p.label) + ")");
			System.out.println(sb.toString());
		} else {
			sb.append(" {" + attributes.get(p.attribute) + "?}");
			System.out.println(sb.toString());
			for(DecTreeNode child: p.children) {
				printTreeNode(child, p, k+1);
			}
		}
	}

	@Override
	public void rootInfoGain(DataSet train) {
		//Initialise the values
		this.labels = train.labels;
		this.attributes = train.attributes;
		this.attributeValues = train.attributeValues;

		//Initialise our infoGainArray to store all the info gain values
		this.infoGainArray = new ArrayList<Double>();
		//Build the tree and store it in our root variable
		this.root = DecisionTreeLearning(train.instances, attributes, 
														train.instances, -1);
		//importance is called to fill up the infoGainArray, as this is the 
		//method that calculates them
		importance(attributes, train.instances);
		//Print the info gains in the desired format
		for(String attri: attributes){
			System.out.format(attri + " %.5f\n", 
					infoGainArray.get(attributeIndex(attri)));
		}
	}
}