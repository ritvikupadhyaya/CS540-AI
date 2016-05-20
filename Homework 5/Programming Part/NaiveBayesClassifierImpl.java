import java.util.*;

/**
 * Your implementation of a naive bayes classifier. Please implement all four 
 * methods.
 */

/**
 * @author ritvik
 *
 */
public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {

	//kDel is the constant delta that is used for smoothing of the probability
	//of word given a label
	double kDel = 0.00001;

	//totalVocabCount will hold the vocabulary size. It is used primarily for
	//smoothing the probability of word given a label
	int totalVocabCount;

	//spamInstanceCount will hold the total number of instances that were spam
	int spamInstanceCount;

	//hamInstanceCount will hold the total number of instances that were ham
	int hamInstanceCount;

	//The lists wordsList and wordsLabelCount are made to store all the unique
	//words we will come across and the labels associated with them respectively
	private List<String> wordsList;
	//Both the lists are modified to match the index for each word. The second
	//list will hold an array of length 2. The first Integer will hold the count
	//of the times the corresponding word was part of ham and second will hold 
	//the number of times the word was part of a spam instance. 
	private List<Integer[]> wordsLabelCount;



	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {

		//Initialize the fields of the class
		this.wordsList = new ArrayList<String>();
		this.wordsLabelCount = new ArrayList<Integer[]>();
		this.totalVocabCount = v;

		/*
		 *This loop goes over all the instances provided to us as part of the 
		 *training. In each pass through the instance, we will update the
		 *required values of the class which will be explained with more detail
		 *when and where they are modified.  
		 */
		for(Instance instance: trainingData){
			//For the purpose of the entire project, I will use an int to store
			//the type of label
			//0 for Label.HAM
			//1 for Label.SPAM
			int labelType = -1;//Default value is -1
			//If the instance was labeled HAM
			if(instance.label == Label.HAM){
				//Increment the total number of ham instances seen so far
				hamInstanceCount++;
				//Update the variable for type of label to 0 (HAM)
				labelType = 0;
			}
			else{//If the instance was labeled SPAM
				//Increment the total number of ham instances seen so far
				spamInstanceCount++;
				//Update the variable for type of label to 0 (SPAM)
				labelType = 1;
			}
			//Iterate over all the words in the instance and update the required
			//variables to reflect the kind of label associated with each word.
			for(String currWord: instance.words){
				//If the word has been previously been added
				if(wordsList.contains(currWord)){
					//wordIndex is the common index for both the wordList and the 
					//wordsLabelCount lists.
					int wordIndex = -1;
					//Get the index of the word from the wordList
					wordIndex = wordsList.indexOf(currWord);
					//Get the number of times the pre-existing word has been 
					//flagged as ham or spam.
					Integer[] updateWLC = wordsLabelCount.get(wordIndex);
					//If the label type of this word's current instance is ham
					if(labelType == 0){//Ham
						//Increment the number of times this word has been ham
						updateWLC[0]++;
					}
					//If the label type of this word's current instance is spam
					else if(labelType == 1){//Spam
						//Increment the number of times this word has been spam
						updateWLC[1]++;
					}
					//Update the values of the word's label types at the index
					//specified
					wordsLabelCount.set(wordIndex, updateWLC);
				}
				//If the word did not exist in the vocabulary before
				else{
					//Add the word to the word list
					wordsList.add(currWord);
					//Start a new count of ham and spam associated with the word
					//which will be 0 initially for either of the two types of
					//label.
					Integer[] newWLC = {0,0};
					//Update the label's count corresponding to the word based
					//on the type of label for the current instance.
					newWLC[labelType]++;
					//Add the label count details to the list. It will be added
					//at the same index as the word list. This is because both 
					//lists are added to 
					wordsLabelCount.add(newWLC);
				}
			}
		}
		//		printSet();
	}

	//Testing method to print out what all words have been added and the times
	//they were classified as HAM or SPAM.
	public void printSet(){
		for(String word: wordsList){
			int index = wordsList.indexOf(word);
			Integer[] counts = wordsLabelCount.get(index); 
			System.out.println("Word"+index+"--"+ word+"--HAM--"+counts[0]+
					"--SPAM--"+counts[1]);
		}
	}
	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {

		//We get the SPAM probability by dividing the total number of SPAM
		//instances with the total number of instances we have seen so far.
		//This will be the sum of instances that were SPAM and HAM.
		double p_l = (spamInstanceCount+0.0)/
				(spamInstanceCount+hamInstanceCount);

		//If the probability needed is for SPAM, then just return the value we
		//calculated. If not then we use the formula P(SPAM) + P(HAM) = 1.
		//Then if the probability is needed for HAM labels, it will be 1 minus
		//the probability we calculated.
		return (label == Label.SPAM)? p_l: (1.0 - p_l);
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
		//p_w_l variable will store the probability of the word given the label
		double p_w_l = 0.0;

		//labelType 		  will hold the label type we were given
		//totalLabelTypeCount stores the number of times the label has been
		//					  associated with all the words we have seen so far. 
		//					  It is sigma of c_l(v)
		//wordTypeCount		  will hold the number of times the specific label
		//					  was associated with the given word. c_l(w)
		int labelType = -1, totalLabelTypeCount = 0, wordTypeCount = 0;

		//Update the labelType based on the label given to us.
		if(label == Label.HAM)
			labelType = 0;
		else
			labelType = 1;

		//Get the total sum of all the times the given label has been associated
		//with the entire vocabulary.
		totalLabelTypeCount = totalLabelsCount(labelType);

		//If the word given exists in the vocabulary we built.
		if(wordsList.contains(word)){
			//Get the index of the word.
			int wordIndex = wordsList.indexOf(word);
			//Update the wordTypeCount to reflect number of times the given word
			//has been associated with the given label.
			wordTypeCount = wordsLabelCount.get(wordIndex)[labelType];
		}

		//If the word did not exist, the variables totalLabelTypeCount, and 
		//wordTypeCount should remain 0 according to the formula given under
		//smoothing in the specification file.
		
		//To smoothen the probability, we will add the delta specified to the
		//numerator and add delta times total vocabulary size to the denominator
		p_w_l = (wordTypeCount + kDel)/
				((totalVocabCount * kDel) + totalLabelTypeCount);

		//Return the calculated value
		return p_w_l;
	}
	
	
	/**
	 * This method calculates the total number of times the specified label
	 * has been associated with all the words in the existing vocabulary.
	 * @param labelType the type of label which we want to find the count for.
	 * 					0 for HAM and 1 for SPAM
	 * @return total number of times the label has been associated with any word
	 * 				 in the vocabulary
	 */
	public int totalLabelsCount(int labelType){
		//Holds the total count
		int returnSum = 0;
		//Goes over the entire wordsLabelCount list
		for(Integer[] set: wordsLabelCount){
			//Add the label type's count to the sum
			returnSum+=set[labelType];
		}
		//return the total count.
		return returnSum;
	}
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {

		//Create an object of the ClassifyResult class to return
		ClassifyResult classResult = new ClassifyResult();

		//We initialize the log of the probabilities of the two types of labels
		//with the log of the probability of the label. Since P(l) will be added
		//to the values anyway.
		double log_prob_spam = Math.log(p_l(Label.SPAM));
		double log_prob_ham = Math.log(p_l(Label.HAM));

		//Go over all the words in the list to calulate the P(w_i|l)
		for(String word: words){
			//Get the probability for the word given the label for both the 
			//labels and add it to the running sum.
			log_prob_spam += Math.log(p_w_given_l(word, Label.SPAM));
			log_prob_ham += Math.log(p_w_given_l(word, Label.HAM));
		}
		
		//Update the internal values of the ClassifyResult object
		classResult.log_prob_ham = log_prob_ham;
		classResult.log_prob_spam = log_prob_spam;
		//Set SPAM as the default label for the return object
		classResult.label = Label.SPAM;
		//But if the probability for HAM was greater, then set HAM as the label
		//for the classification
		if(log_prob_ham > log_prob_spam)
			classResult.label = Label.HAM;
		//Return the classification
		return classResult;
	}
}
