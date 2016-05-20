/*
 * Ritvik Upadhyaya
 * CS540 HW3 question 3
 * KMeans.java 
 */
import java.util.ArrayList;
public class KMeans 
{
	//Calulates the euclidean distance between the given centroid and the given
	//instance over all their dimentions.
	private double euclideanDistance(double[] instance, double[] centroid)
	{
		//Hold the sum of the square of the difference between dimention values 
		double distSquared = 0;
		//go over all the dimentions of the centroid and instance
		for(int i = 0; i < centroid.length; i++)
		{
			distSquared += ((instance[i] - centroid[i])
					*
					(instance[i] - centroid[i]));
		}
		//Return the square root of the sum. 
		return Math.sqrt(distSquared);
	}

	//Method averages the instances for a given centroid at the centroidIndex.
	//We get the dimension of the centroid we are looking at along with the 
	//complete list of instances and the clusterAssignment.
	private double centroidAverage(int centroidIndex, int dimension, 
			double[][] instances, int[] clusterAssignment){
		int instanceNum = 0;//Number of instances in centroid
		double averageSum = 0;//Sum of values of all instances in centroid
		//Go over all the instances in the clusterAssignment
		for(int i = 0; i < instances.length; i++){
			//Check if the instance is a part of the centroid being considered
			if(clusterAssignment[i] == centroidIndex){
				//Update values
				averageSum += instances[i][dimension];
				instanceNum++;
			}
		}
		//Return average
		return averageSum/instanceNum;
	}

	//allocClusters assigns initializes/assigns centroids to each instance given
	//It takes in all the instances and all the centroids given to us and the
	//current clusterAssignment and returns the final clusterAssignment after
	//all the instances are allocated a centroid.
	private int[] allocClusters(double[][] instances,
			double[][] centroids, int[] clusterAssignment) {

		//Go over all the instances
		for(int i = 0; i < instances.length; i++){
			int centroidIndex = 0;

			//We need a baseline distance that holds the closest centroid to the
			//instnace's distance.
			//The centroidToCurrInstanceDist stores the distance between the 
			//instance and the centroid being considered
			double minDist = euclideanDistance(instances[i],centroids[0]), 
					centroidToCurrInstanceDist = 0;
			//Match up all the centroid distances to the current centroid
			for(int j = 1; j < centroids.length; j++)
			{
				//Update the current distance
				centroidToCurrInstanceDist = 
						euclideanDistance(instances[i],centroids[j]);

				//If a lower distance is found then we update the minDist and
				//update the corresponding centroid index.
				if(centroidToCurrInstanceDist < minDist)
				{
					minDist = centroidToCurrInstanceDist;
					centroidIndex = j;
				}
			}

			//Assign the instance to the lowest found distance centroid's index
			//Update the clusterAssignment
			clusterAssignment[i] = centroidIndex;
		}
		return clusterAssignment;
	}

	//This method deals with the case when we find a cluster is empty.
	//The method takes in the index of the empty centroid,
	//all the centroids, the instances, and the current clusterAssignment to 
	//first re-evaluate the distances and then reassign instances to the given 
	//centroid at the empty index.
	public int[] allocOrphanClusters(int emptyIndex,double[][] centroids, 
			double[][] instances,int[] clusterAssignment)
	{
		//maxDist is the baseline distnace of the instance from centroid
		//emptyToCurrInstanceDist is the current (in iteration) instance's
		//distance to the empty centroid.
		double maxDist = 0, emptyToCurrInstanceDist = 0;
		int lastMatchedInstanceIndex = 0;

		//We set the baseline from first instance to the empty centroid
		maxDist = euclideanDistance(centroids[emptyIndex], instances[0]);

		//Go through all the instances
		for(int i = 1; i < instances.length; i++)
		{
			//Update the emptyToCurrInstanceDist from current instance to the
			//empty centroid
			emptyToCurrInstanceDist = euclideanDistance(centroids[emptyIndex],
					instances[i]);

			//If the current instance is farther away then get its index
			if(emptyToCurrInstanceDist > maxDist)
			{
				//keep track of the farthest instance x
				maxDist = emptyToCurrInstanceDist;
				lastMatchedInstanceIndex = i;
			}
		}

		//Then set the empty centroid to contain the index.
		//Update the clusterAssignment accordingly
		clusterAssignment[lastMatchedInstanceIndex] = emptyIndex;
		return clusterAssignment;
	}

	//Method goes through a list of clusterAssignment for all the instances
	//checks if there is any centroid that was not assigned an instance.
	//Method takes in the total centroids present and the current 
	//ClusterAssignment
	public int getEmptyClusterIndex(int[] clusterAssignment, int totalCentroids)
	{
		//To keep track of finding any empty centroids
		boolean found;
		//Loop counter
		int i = 0;
		//Go over all the centroids and see that they contain at least one 
		//instance
		while (i < totalCentroids){
			//Have not found anything at the start of each centroid search
			found = false;
			//Check if any instance in clusterAssignment has this centroid
			for(int j = 0; j < clusterAssignment.length; j++)
			{
				//If the centroid is found at one value
				if(i == clusterAssignment[j])
				{
					//Then we do not need to find another match. Move on to the
					//next centroid if any.
					found = true;
					break;
				}
			}
			//If nothing was found in the centroid then return the index of the
			//empty centroid
			if(!found){
				return i;
			}
			//else move on the next centroid in loop
			i++;
		}

		//If no centroid is empty then return -1
		return -1;
	}

	//Method takes in the distortion iteration list, all the centroids, all the 
	//instances, the threshold given to us, and the current clusterAssignment.
	//It then calculated the distortion and then compares the threshold from 
	//that to the given threshold. Returns true if calculated threshold is
	//greater than the given threshold. Else returns false.
	public boolean distortionThresholdReached(ArrayList<Double> 
		shadowDistortionIterations, double[][] centroids, 
		double[][] instances, 
		double givenThreshold, int[] clusterAssignment)
	{

		double calculatedDistortion = 0; //The variable stores the distortion we
		//will calculate
		int numOfIterations = 0;

		//Going over all the instances
		for(int i = 0; i < instances.length; i++)
		{
			//And all the dimensions of the instance 
			for(int j = 0; j < centroids[0].length; j++)
			{
				//Add the calculated value to the total distortion
				calculatedDistortion += ((instances[i][j] 
						- centroids[clusterAssignment[i]][j])
						*
						(instances[i][j] 
								- centroids[clusterAssignment[i]][j])); 
			}
		}
		//Add the calculatedDistortion to the iteration list
		shadowDistortionIterations.add(calculatedDistortion);

		//Now we will calculate our threshold and compare it to the given
		//threshold
		//See the number of iterations
		numOfIterations = shadowDistortionIterations.size();

		//We need to have at least 2 iterations because we calculate threshold
		//after each successive step
		if(numOfIterations > 1)
		{
			double calculatedThreshold;
			//calculate current threshold
			calculatedThreshold = 
					shadowDistortionIterations.get(numOfIterations - 1) 
					- shadowDistortionIterations.get(numOfIterations - 2);

			//If the threshold is 0 we will break out of the main loop. 
			if(calculatedThreshold == 0 || 
					shadowDistortionIterations.get(numOfIterations - 2) == 0)
				return true;

			//Get the absolute value of the threshold being calculated
			calculatedThreshold = Math.abs((calculatedThreshold / 
					shadowDistortionIterations.get(numOfIterations - 2)));

			//if threshold was met
			if(calculatedThreshold < givenThreshold)
				return true;
		}
		//threshold was not met
		return false;
	}

	public KMeansResult cluster(double[][] centroids, 
			double[][] instances, 
			double threshold) 
	{
		ArrayList<Double> shadowDistortionIterations = new ArrayList<Double>();
		int[] clusterAssignment = new int[instances.length];
		//Main Loop Condition
		//Loop should not get over till the threshold condition is not met
		while(true)
		{
			int emptyIndex = -2;//Stores the index of the empty centroid and is
			//-1 when there is no empty centroid.
			//Assign instance to centroids
			//Checks if any cluster is empty, if so then reassigns till it finds
			//a distribution without any empty cluster
			while(emptyIndex != -1)
			{
				//Store the new allocation in the cluster assignment array
				clusterAssignment = allocClusters(instances, centroids, 
						clusterAssignment);

				//Get the index of the empty centroid. If non is found, -1. 
				emptyIndex = getEmptyClusterIndex(clusterAssignment, 
						centroids.length);

				//Is there an empty cluster
				if(emptyIndex !=-1)
				{
					//If empty cluster is found, instances are reassigned.
					clusterAssignment=allocOrphanClusters(emptyIndex,centroids, 
							instances,clusterAssignment);
				}
			}

			//Update the centroids to according to the average location of all 
			//the instances.
			//For all the centroids
			for(int i = 0; i < centroids.length; i++){
				//Go over all the instances assigned to the centroid
				for(int j = 0; j < centroids[i].length; j++)
				{
					//Update the coordinates based on the average distance of 
					//all the centroids
					centroids[i][j] = centroidAverage(i,j, instances, 
							clusterAssignment);
				}
			}

			//Check if we have reached the threshold based on distortion 
			//calculations
			//if we have break out of the main loop
			if(distortionThresholdReached(shadowDistortionIterations, centroids,
					instances, threshold, clusterAssignment))
			{
				//KMeans was successful
				break;
			}
			//We have not crossed the threshold.
			else
			{
				//Main loop has to go again
				//Reassign the clusters and update the new clusterAssignment 
				//array
				clusterAssignment = allocClusters(instances, centroids, 
						clusterAssignment);
			}
		}
		//When we are done assigning clusters

		//Make the result object to return and update its values
		KMeansResult result = new KMeansResult();

		result.centroids = centroids; //Same centroids as before
		result.distortionIterations = new double
				[shadowDistortionIterations.size()]; //The list of distortion
		//iterations.

		for(int i = 0; i < result.distortionIterations.length; i++)
		{
			result.distortionIterations[i] = shadowDistortionIterations.get(i);
		}
		//The final clusterAssignment
		result.clusterAssignment = clusterAssignment;

		return result;
	}
}