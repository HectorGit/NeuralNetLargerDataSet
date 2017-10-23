/*
==============================================================================

NeuralNetwork.cpp
Created: 29 Sep 2017 2:54:03pm
Author:  Owner

//VERSION 1 OF CODE BASED ON ALEX THOMO'S IMPLEMENTATION

==============================================================================
*/
#include "stdafx.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>
#include <math.h> //or cmath?
#include "NeuralNetwork.h"

/*
NOTES: MAKE IT MORE EXPLICIT WHAT EACH FUNCTION IS TAKING IN AND RETURNING IF YOU CAN?!!
*/

//Constructor
NeuralNetwork::NeuralNetwork(std::vector<int> &paramSizes)
{
	bool debug = false;
	//printf("what is going on?");
	numLayers = paramSizes.size();
	mini_batch_size = 10;
	if (debug) { printf("\n numLayers is: %d \n", numLayers); }

	eta = 3.0; //should pass this in to SGD --> CHECK... DONT SET IT HERE!
	sizes = paramSizes; //NOT SURE IF THIS WORKS //MIGHT NEED TO COPY IT IN SOMEHOW.

	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n------------Initializing NeuralNet------------\n"); }

	if (debug) { printf("\nLEVEL ONE\n"); }

	weightsMatrixL1 = Eigen::MatrixXf::Random(sizes[1], sizes[0]); //check this //would be a 5 by 3 matrix
	if (debug) {
		printf("\nHere's weightsMatrixL1\n");
		std::cout << weightsMatrixL1.format(CleanFmt);
	}
	gradientWsL1 = Eigen::MatrixXf::Zero(sizes[1], sizes[0]); //5 by 3 matrix
	if (debug) {
		printf("\nHere's gradientWsL1\n");
		std::cout << gradientWsL1.format(CleanFmt);
	}

	biasesMatrixL1 = Eigen::MatrixXf::Random(sizes[1], 1); // 5 by 1 matrix
	if (debug) {
		printf("\nHere's biasesMatrixL1\n");
		std::cout << biasesMatrixL1.format(CleanFmt);
	}
	gradientBsL1 = Eigen::MatrixXf::Zero(sizes[1], 1);    //5 by 1 matrix
	if (debug) {
		printf("\nHere's gradientBsL1\n");
		std::cout << gradientBsL1.format(CleanFmt);
	}

	if (debug) { printf("\n\nLEVEL TWO\n"); }

	weightsMatrixL2 = Eigen::MatrixXf::Random(sizes[2], sizes[1]); //3 by 2 matrix
	if (debug) {
		printf("\nHere's weightsMatrixL2\n");
		std::cout << weightsMatrixL2.format(CleanFmt);
	}
	gradientWsL2 = Eigen::MatrixXf::Zero(sizes[2], sizes[1]);    //3 by 2 matrix
	if (debug) {
		printf("\nHere's gradientWsL2\n");
		std::cout << gradientWsL2.format(CleanFmt);
	}

	biasesMatrixL2 = Eigen::MatrixXf::Random(sizes[2], 1);      //3 by 1 matrix
	if (debug) {
		printf("\nHere's biasesMatrixL2\n");
		std::cout << biasesMatrixL2.format(CleanFmt);
	}
	gradientBsL2 = Eigen::MatrixXf::Zero(sizes[2], 1);         //3 by 1 matrix
	if (debug) {
		printf("\nHere's gradientBsL2\n");
		std::cout << gradientBsL2.format(CleanFmt);
	}

	if (debug) { printf("\n --------DONE Initializing NeuralNet------- \n"); }

}

NeuralNetwork::~NeuralNetwork()
{
	//release all memory or something Lolz
}

void NeuralNetwork::feedForward(Eigen::MatrixXf &x, Eigen::MatrixXf &y)
{
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) {
		printf("--------feedforward algorithm------");
		printf("\n----------LEVEL ONE ---------\n");
	}

	//this weightsMatrix has been previously modified by the last fed mini batch.
	Eigen::MatrixXf dotProductL1 = weightsMatrixL1 * x;
	if (debug) {
		printf("\n wT dot x \n");
		std::cout << "\n w: \n" << weightsMatrixL1.format(CleanFmt);
		std::cout << "\n x: \n" << x.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	//zL1 at this point contains the result of the previous fed mini batch.
	//gets overwritten
	zL1 = dotProductL1 + biasesMatrixL1;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL1.format(CleanFmt);
		std::cout << "\n Result: \n" << zL1.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	//activation 1 has the result of the previous fed mini batch.
	activationL1 = sigmoid_Vectorial(zL1);
	if (debug) {
		printf("\n\n Apply Sigmoid\n");
		std::cout << "\n Result: \n" << activationL1.format(CleanFmt);
	}

	if (debug) { printf("\n\n---------LEVEL TWO ----------\n"); }

	Eigen::MatrixXf dotProductL2 = weightsMatrixL2*activationL1;
	if (debug) {
		printf("\n wT dot activationL1 \n");
		std::cout << "\n w: \n" << weightsMatrixL2.format(CleanFmt);
		std::cout << "\n activationL1: \n" << activationL1.format(CleanFmt);
		std::cout << "\n Result: \n" << dotProductL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	//zL2 had the proceeds of the last mini batch- careful! - gets overwritten?
	zL2 = dotProductL2 + biasesMatrixL2;
	if (debug) {
		printf("\n + Biases \n");
		std::cout << "\n b: \n" << biasesMatrixL2.format(CleanFmt);
		std::cout << "\n Result: \n" << zL2.format(CleanFmt);
	}
	//---------------------------------------------------------------------------------
	//activ L2 also has proceeds from last fed mini batch
	// but it gets overwritten.
	activationL2 = sigmoid_Vectorial(zL2);
	if (debug) {
		printf("\n\n Apply Sigmoid\n");
		std::cout << "\n Result: \n" << activationL2.format(CleanFmt);
	}
	if (debug) { printf("\n\n---------DONE----------\n"); }

}

/*Requires FeedForward to be working*/ /*called after an 'epoch' ends*/
									   /*do we really need testData here??*/
int NeuralNetwork::evaluate(Eigen::MatrixXf &activationL2/*Eigen::MatrixXf &testData*/, Eigen::MatrixXi &y)
{
	/*
	in this implementation it just acts as evaluating
	a single row of outputs w float values versus a int-valued 1D matrix
	fix when the time comes.
	*/

	bool debug = false;

	int counterOfMatches = 0;

	Eigen::MatrixXi::Index classificationIndex;
	Eigen::MatrixXf maxVal = Eigen::MatrixXf::Zero(10, 1);

	for (int m = 0; m< activationL2.rows(); m++)
	{
		maxVal(m) = activationL2.row(m).maxCoeff(&classificationIndex);
		if (debug) { printf("\n comparing %d, %d \n", classificationIndex, y(m)); }
		if (classificationIndex == y(m))
		{
			if (debug) { printf("\n match %d, %d \n", classificationIndex, y(m)); }
			counterOfMatches++;
		}
	}
	return counterOfMatches;
}

Eigen::MatrixXf NeuralNetwork::costDerivative(Eigen::MatrixXf &outputActivations, Eigen::MatrixXf &y)
{
	//should this be in a for loop for every instance that came in?? or is it called multiple times?
	return outputActivations - y;
}

Eigen::MatrixXf NeuralNetwork::sigmoid_Vectorial(Eigen::MatrixXf &z)
{
	int r = z.rows();
	int c = z.cols();
	Eigen::MatrixXf returnedMatrix = Eigen::MatrixXf::Zero(r, c);
	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			returnedMatrix.array()(i, j) = 1.0 / (1.0 + exp(-1.0 * z.array()(i, j))); //exp comes from Math.
		}
	}
	return returnedMatrix;
}

Eigen::MatrixXf NeuralNetwork::sigmoid_Prime_Vectorial(Eigen::MatrixXf &z)
{
	int r = z.rows();
	int c = z.cols();
	Eigen::MatrixXf sigmoidMatrix = sigmoid_Vectorial(z);
	Eigen::MatrixXf returnedMatrix = Eigen::MatrixXf::Zero(r, c);
	returnedMatrix = sigmoidMatrix.array()*(1 - sigmoidMatrix.array());
	return returnedMatrix;
}

void NeuralNetwork::stochasticGradientDescent(/*Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData*/)
{
	/*
	implement a for loop to select mini batches OR
	it looks like we could put this into a for loop
	which grabs all_Xs[] from k to k+miniBatchSize.
	it also grabs corresponding all_Ys[] from k to k_miniBatchSize.
	it will check if there are still .

	*/
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");


	if (debug) { printf("\n stochastic gradient descent was called \n"); }

	int numberOfMiniBatches = all_Xs.size() / mini_batch_size; // this better be an INT! - need multiple - see main.cpp
	if (debug) { printf("\n numberofMiniBatches, %d \n", numberOfMiniBatches); }

	int k = mini_batch_size;
	epochs = numberOfMiniBatches;

	for (int i = 0; i < epochs; i++) {

		//before running for that epoch,
		//we select the mini batches and put them
		//in mini_Batch_Xs and  mini_Batch_Ys
		for (int m = i*k; m < (i + 1)*k; m++) {
			mini_Batch_Xs.emplace_back(all_Xs[m]);
			mini_Batch_Ys.emplace_back(all_Ys[m]);
		}

		if (debug) {
			printf("\n --------MINI BATCH Xs %d -------- \n", i);
			for (int j = 0; j < k; j++) {
				std::cout << "\n" << mini_Batch_Xs[j].format(CleanFmt) << "\n";
				printf("\n");
			}
			printf("\n --------MINI BATCH Ys %d -------- \n", i);
			for (int j = 0; j < k; j++) {
				std::cout << "\n" << mini_Batch_Ys[j].format(CleanFmt) << "\n";
				printf("\n");
			}
		}
		
		if (debug) { printf("\n epoch [%d] \n", i); }
		updateMiniBatch();

	}

	printf("\n Done Stochastic Gradient Descent \n");
}

void NeuralNetwork::updateMiniBatch(/*Eigen::MatrixXf &mini_batch,*/ /*float eta*//*Eigen::MatrixXf &biasesMatrixL1,*/ /*Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2*/)
{
	/*we have defined the following matrices already*/
	/*LEVEL 1
	Eigen::MatrixXf weightsMatrixL1; 3 by 2 filled w current ws
	Eigen::MatrixXf gradientWsL1;    3 by 2 filled with zeros now
	Eigen::MatrixXf biasesMatrixL1;  3 by 1 filled w current bs
	Eigen::MatrixXf gradientBsL1;    3 by 1 filled with zeros now*/

	/*LEVEL 2
	Eigen::MatrixXf weightsMatrixL2; 2 by 3  filled w current ws
	Eigen::MatrixXf gradientWsL2;    2 by 3  filled with zeros now
	Eigen::MatrixXf biasesMatrixL2;  2 by 1  filled w current bs
	Eigen::MatrixXf gradientBsL2;    2 by 1  filled with zeros now */

	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	if (debug) { printf("\n update mini batch was called \n"); }

	if (debug) {
		printf("\n old weights and biases \n");
		std::cout << "\n weightsMatrixL1 \n" << weightsMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n weightsMatrixL2 \n" << weightsMatrixL2.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL1 \n" << biasesMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL2 \n" << biasesMatrixL2.format(CleanFmt) << "\n";
	}


	//need to move what is in main to here for the moment. (DUMMY STUFF)
	backPropagation(mini_batch_size);

	//1. accumulate all the gradients.
	for (int i = 0; i < mini_batch_size; i++) {
		/*if (debug) {
		printf("\n gradientWsL1 rows = %d cols = %d \n", gradientWsL1.rows(), gradientWsL1.cols());
		std::cout << "\n" << gradientWsL1.format(CleanFmt) << "\n";

		printf("\n allGradientsWs[%d][0] rows = %d cols = %d \n", i, allGradientsWs[i][0].rows(), allGradientsWs[i][0].cols());
		std::cout << "\n" << allGradientsWs[i][0].format(CleanFmt) << "\n";
		}*/

		gradientWsL1 = gradientWsL1 + allGradientsWs[i][0];
		gradientWsL2 = gradientWsL2 + allGradientsWs[i][1];
		gradientBsL1 = gradientBsL1 + allGradientsBs[i][0];
		gradientBsL2 = gradientBsL2 + allGradientsBs[i][1];
	}


	//2. get an average (+other operations)
	if (debug) { printf("\n value of the multiplier for gradient matrices = %4.2f ", eta / (float)mini_batch_size); }
	weightsMatrixL1 = weightsMatrixL1 - ((eta / (float)mini_batch_size) * gradientWsL1);
	weightsMatrixL2 = weightsMatrixL2 - ((eta / (float)mini_batch_size) * gradientWsL2);
	biasesMatrixL1 = biasesMatrixL1 - ((eta / (float)mini_batch_size) * gradientBsL1);
	biasesMatrixL2 = biasesMatrixL2 - ((eta / (float)mini_batch_size) * gradientBsL2);


	if (debug) {
		printf("\n new weights and biases \n");
		std::cout << "\n weightsMatrixL1 \n" << weightsMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n weightsMatrixL2 \n" << weightsMatrixL2.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL1 \n" << biasesMatrixL1.format(CleanFmt) << "\n";
		std::cout << "\n biasesMatrixL2 \n" << biasesMatrixL2.format(CleanFmt) << "\n";
	}
}

void NeuralNetwork::backPropagation(int mini_batch_size/*, mini_Batch_Xs, mini_Batch_Ys*/)
{
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	/*
	these became class members.
	Eigen::MatrixXf allGradientsBs[10][2]; //nabla_b
	Eigen::MatrixXf allGradientsWs[10][2]; //nabla_w
	Eigen::MatrixXf allActivations[10][3];
	Eigen::MatrixXf allZs[10][2];
	*/

	//STEP 1 - 
	for (int i = 0; i < mini_batch_size; i++) {
		if (debug) { printf("\n feedforward for mini_Batch_Xs[%d] \n", i); }
		
		activationL1 = Eigen::MatrixXf::Zero(activationL1.rows(), activationL1.cols());
		activationL2 = Eigen::MatrixXf::Zero(activationL2.rows(), activationL2.cols());
		zL1 = Eigen::MatrixXf::Zero(zL1.rows(), zL1.cols());
		zL2 = Eigen::MatrixXf::Zero(zL2.rows(), zL2.cols());
		
		feedForward(mini_Batch_Xs[i], mini_Batch_Ys[i]); //does this work?

		allActivations[i][0] = mini_Batch_Xs[i];
		allActivations[i][1] = activationL1;
		allActivations[i][2] = activationL2;

		allZs[i][0] = zL1;
		allZs[i][1] = zL2;

		//we could zero out the activations here so they don't affect the next iteration,
		//have suspicion that remaining stuff is bothering the procedure.

		//activation L1 = matrix of zeros and activation L2 too.
		//we could also empty the zL1 and zL2 for the next feedforward.
		
	}

	if (debug) {
		printf("\n starting \n");
		for (int i = 0; i < mini_batch_size; i++) {

			printf("\n showing allActivations for single_x # %d \n", i);
			for (int k = 0; k < numLayers; k++) {
				Eigen::MatrixXf temp = allActivations[i][k];
				std::cout << "\n" << temp.format(CleanFmt) << "\n";
			}

			printf("\n showing all_Zs for single_x # %d \n", i);
			for (int j = 0; j < numLayers - 1; j++) {
				Eigen::MatrixXf temp = allZs[i][j];
				std::cout << "\n" << temp.format(CleanFmt) << "\n";
			}
		}
		printf("\n done \n");
	}
	//at this point we have a matrix of z's and matrix of activations.

	//-------------BackWardsPass------------------

	//looks like somewhat repetitive code, but didn't know how else to do it.
	//draw it by hand to check dimensions expected.

	if (debug) { printf("\n BACKWARDS PROPAGATION \n"); }

	//start with populating level 2 by using the error
	Eigen::MatrixXf delta[10];
	for (int i = 0; i < mini_batch_size; i++) {

		if (debug) {
			printf("\n allActivations[%d][2] row = %d   col = %d \n", i, allActivations[i][2].rows(), allActivations[i][2].cols());
			std::cout << mini_Batch_Ys[i].format(CleanFmt);
			printf("\n allZs[%d][1] row = %d   col = %d \n", i, allZs[i][1].rows(), allZs[i][1].cols());
		}

		//SINGLE deltaL2 = (activationL2 - y) times sigmoid zL2 - should have dimensions as BSmatrix 2 by 1
		delta[i] = (costDerivative(allActivations[i][2], mini_Batch_Ys[i])).cwiseProduct(sigmoid_Prime_Vectorial(allZs[i][1]));
	}
	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n delta[%d] : \n", i);
			std::cout << delta[i].format(CleanFmt) << "\n";
		}
	}

	for (int i = 0; i < mini_batch_size; i++) { //also down here 2 is the last level
		allGradientsBs[i][1] = delta[i]; //cause 2 is the last level.
		if (debug) { printf("\nallActivations[%d][1].transpose() rows = %d col = %d \n", i, allActivations[i][1].transpose().rows(), allActivations[i][1].transpose().cols()); }
		allGradientsWs[i][1] = delta[i] * allActivations[i][1].transpose();
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n allGradientsBs[%d][1] : \n", i);
			std::cout << allGradientsBs[i][1].format(CleanFmt) << "\n";
			printf("\n allGradientsWs[%d][1] : \n", i);
			std::cout << allGradientsWs[i][1].format(CleanFmt) << "\n";
		}
	}

	//populate level 1 by using the error as well
	Eigen::MatrixXf deltaL1[10];

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			//this is supposed to require transposition?!
			printf("\n weightsMatrixL2.transpose() rows = %d cols = %d \n", weightsMatrixL2.transpose().rows(), weightsMatrixL2.transpose().cols());
			std::cout << deltaL1[i].format(CleanFmt) << "\n";
			printf("\n delta[%d] \n", i);// IS THIS RIGHT?
			std::cout << delta[i].format(CleanFmt) << "\n"; //IS THIS RIGHT???
			printf("\n allActivations[%d][0] rows = %d cols = %d \n", i, allActivations[i][0].rows(), allActivations[i][0].cols());
			std::cout << allActivations[i][0].format(CleanFmt) << "\n";
		}
	}

	for (int i = 0; i < mini_batch_size; i++) {

		deltaL1[i] = (weightsMatrixL2.transpose() * delta[i]).cwiseProduct(allActivations[i][1]);
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n deltaL1[%d] : \n", i);
			std::cout << deltaL1[i].format(CleanFmt) << "\n";
		}
	}

	for (int i = 0; i < mini_batch_size; i++) { //also down here 2 is the last level
		allGradientsBs[i][0] = deltaL1[i];
		allGradientsWs[i][0] = deltaL1[i] * allActivations[i][0].transpose();
	}

	if (debug) {
		for (int i = 0; i<mini_batch_size; i++) {
			printf("\n allGradientsBs[%d][0] rows = %d cols = %d \n", i, allGradientsBs[i][0].rows(), allGradientsBs[i][0].cols());
			std::cout << allGradientsBs[i][0].format(CleanFmt) << "\n";
			printf("\n allGradientsWs[%d][0] rows = %d cols = %d \n", i, allGradientsWs[i][0].rows(), allGradientsWs[i][0].cols());
			std::cout << allGradientsWs[i][0].format(CleanFmt) << "\n";
		}
	}
}

// 'classify' will choose the class a vector would be in - 1 , 2 or 3.
// similar to 'classification index' from within evaluate() - 
void NeuralNetwork::classify(int numberOfAttributes, std::vector<int> &paramSizes/*technically should receive a std::vector of Matrix, but it's hard so avoided passing it*/)
{
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	//int correct1s, onesas2s, onesas3s = 0;
	//int twosas1s, correct2s, twosas3s = 0;
	//int threesas1s, threessas2s, correct3s = 0;
	Eigen::MatrixXf counters = Eigen::MatrixXf::Zero(paramSizes[2], paramSizes[2]); // how to make this more dynamic.

	Eigen::MatrixXf classifications = Eigen::MatrixXf::Zero(dummyTestDataSet_Xs.size(), 1);
	Eigen::MatrixXf yS = Eigen::MatrixXf::Zero(dummyTestDataSet_Ys.size(), 1);

	for (int i = 0; i < dummyTestDataSet_Xs.size(); i++) {
		//remember calling feedforward overwrites all previous stuff written in matrices !!!

		if (debug) {
			std::cout << "\n dummyTestDataSet_Xs[i].format(CleanFmt)\n" << dummyTestDataSet_Xs[i].format(CleanFmt) << "\n";
			std::cout << "\n dummyTestDataSet_Ys[i].format(CleanFmt)\n" << dummyTestDataSet_Ys[i].format(CleanFmt) << "\n";
		}

		//SO MY GUESS IS THAT HERE WE SHOULD CLEAR EVERYTHING THAT WAS PREVIOUSLY IN THE activations.

		activationL1 = Eigen::MatrixXf::Zero(activationL1.rows(), activationL1.cols());
		activationL2 = Eigen::MatrixXf::Zero(activationL2.rows(), activationL2.cols());
		zL1 = Eigen::MatrixXf::Zero(zL1.rows(), zL1.cols());
		zL2 = Eigen::MatrixXf::Zero(zL2.rows(), zL2.cols());

		feedForward(dummyTestDataSet_Xs[i], dummyTestDataSet_Ys[i]); //does this work?


		int maxVal = 0;
		Eigen::MatrixXi::Index highestNumIndex;
		maxVal = activationL2.col(0).maxCoeff(&highestNumIndex);
		if(debug){
			std::cout << "\n\n";
			std::cout << "activation L2 for this instance : \n" << activationL2.format(CleanFmt);
			std::cout << "\n\n"; 
		}
		classifications(i, 0) = highestNumIndex;
		Eigen::MatrixXi::Index indexOfValueExpected;
		maxVal = dummyTestDataSet_Ys[i].col(0).maxCoeff(&indexOfValueExpected);
		yS(i, 0) = indexOfValueExpected;

		countErrors(i, counters, classifications, yS);
	}

	if (debug) {
		std::cout << "\n\n";
		std::cout << "classifications: \n" << classifications.format(CleanFmt);
		std::cout << "\n\n";
		std::cout << "values expected: \n" << yS.format(CleanFmt);
		std::cout << "\n\n";
		std::cout << "\n\n";
		std::cout << "counters: \n" << counters.format(CleanFmt);
		std::cout << "\n\n";
	}

	// call accuracy - pass total and pass correct
	float accuracy = calculateAccuracy(dummyTestDataSet_Xs.size(), counters);

	Eigen::MatrixXf probMatrix = obtainConfusionMatrix(dummyTestDataSet_Xs.size(), counters);

}

//NEED A BETTER WAY TO DO THIS!
//make a for loop or something! dep on the size of Output Vector
void NeuralNetwork::countErrors(int index, Eigen::MatrixXf &counters, Eigen::MatrixXf &classifications, Eigen::MatrixXf &yS) {
	if (yS(index, 0) == 0 && classifications(index, 0) == 0) {
		counters(0, 0) += 1;
	}
	else if (yS(index, 0) == 0 && classifications(index, 0) == 1) {
		counters(0, 1) += 1;
	}
	else if (yS(index, 0) == 0 && classifications(index, 0) == 2) {
		counters(0, 2) += 1;
	}
	else if (yS(index, 0) == 0 && classifications(index, 0) == 3) {
		counters(0, 3) += 1;
	}
	else if (yS(index, 0) == 0 && classifications(index, 0) == 4) {
		counters(0, 4) += 1;
	}
	else if (yS(index, 0) == 0 && classifications(index, 0) == 5) {
		counters(0, 5) += 1;
	}
	///
	if (yS(index, 0) == 1 && classifications(index, 0) == 0) {
		counters(1, 0) += 1;
	}
	else if (yS(index, 0) == 1 && classifications(index, 0) == 1) {
		counters(1, 1) += 1;
	}
	else if (yS(index, 0) == 1 && classifications(index, 0) == 2) {
		counters(1, 2) += 1;
	}
	else if (yS(index, 0) == 1 && classifications(index, 0) == 3) {
		counters(1, 3) += 1;
	}
	else if (yS(index, 0) == 1 && classifications(index, 0) == 4) {
		counters(1, 4) += 1;
	}
	else if (yS(index, 0) == 1 && classifications(index, 0) == 5) {
		counters(1, 5) += 1;
	}
	///
	if (yS(index, 0) == 2 && classifications(index, 0) == 0) {
		counters(2, 0) += 1;
	}
	else if (yS(index, 0) == 2 && classifications(index, 0) == 1) {
		counters(2, 1) += 1;
	}
	else if (yS(index, 0) == 2 && classifications(index, 0) == 2) {
		counters(2, 2) += 1;
	}
	else if (yS(index, 0) == 2 && classifications(index, 0) == 3) {
		counters(2, 3) += 1;
	}
	else if (yS(index, 0) == 2 && classifications(index, 0) == 4) {
		counters(2, 4) += 1;
	}
	else if (yS(index, 0) == 2 && classifications(index, 0) == 5) {
		counters(2, 5) += 1;
	}
	///
	if (yS(index, 0) == 3 && classifications(index, 0) == 0) {
		counters(3, 0) += 1;
	}
	else if (yS(index, 0) == 3 && classifications(index, 0) == 1) {
		counters(3, 1) += 1;
	}
	else if (yS(index, 0) == 3 && classifications(index, 0) == 2) {
		counters(3, 2) += 1;
	}
	else if (yS(index, 0) == 3 && classifications(index, 0) == 3) {
		counters(3, 3) += 1;
	}
	else if (yS(index, 0) == 3 && classifications(index, 0) == 4) {
		counters(3, 4) += 1;
	}
	else if (yS(index, 0) == 3 && classifications(index, 0) == 5) {
		counters(3, 5) += 1;
	}
	///
	if (yS(index, 0) == 4 && classifications(index, 0) == 0) {
		counters(4, 0) += 1;
	}
	else if (yS(index, 0) == 4 && classifications(index, 0) == 1) {
		counters(4, 1) += 1;
	}
	else if (yS(index, 0) == 4 && classifications(index, 0) == 2) {
		counters(4, 2) += 1;
	}
	else if (yS(index, 0) == 4 && classifications(index, 0) == 3) {
		counters(4, 3) += 1;
	}
	else if (yS(index, 0) == 4 && classifications(index, 0) == 4) {
		counters(4, 4) += 1;
	}
	else if (yS(index, 0) == 4 && classifications(index, 0) == 5) {
		counters(4, 5) += 1;
	}
	///
	if (yS(index, 0) == 5 && classifications(index, 0) == 0) {
		counters(5, 0) += 1;
	}
	else if (yS(index, 0) == 5 && classifications(index, 0) == 1) {
		counters(5, 1) += 1;
	}
	else if (yS(index, 0) == 5 && classifications(index, 0) == 2) {
		counters(5, 2) += 1;
	}
	else if (yS(index, 0) == 5 && classifications(index, 0) == 3) {
		counters(5, 3) += 1;
	}
	else if (yS(index, 0) == 5 && classifications(index, 0) == 4) {
		counters(5, 4) += 1;
	}
	else if (yS(index, 0) == 5 && classifications(index, 0) == 5) {
		counters(5, 5) += 1;
	}
	///
}

//vectorial method
//MAKE A FOR LOOP TOO! //depending on the sizeOfTheOutputVector .
float NeuralNetwork::calculateAccuracy(int total, Eigen::MatrixXf &counters)
{
	float accuracy = 0;
	int sum = 0;
	sum += counters(0, 0);
	sum += counters(1, 1);
	sum += counters(2, 2);
	sum += counters(3, 3);
	sum += counters(4, 4);
	sum += counters(5, 5);

	accuracy = ((float)sum / (float)total) * 100.0;
	printf("The accuracy was %4.2f \%", accuracy);
	return accuracy;
}

Eigen::MatrixXf NeuralNetwork::obtainConfusionMatrix(int total, Eigen::MatrixXf &counters)
{
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");

	Eigen::MatrixXf probabilityMatrix = (counters)*(1.0 / (float)total);
	std::cout << "\n\n";
	std::cout << "Confusion Matrix: \n" << counters.format(CleanFmt);
	std::cout << "\n\n";
	std::cout << "\n\n";
	std::cout << "Probability Matrix: \n" << probabilityMatrix.format(CleanFmt);
	std::cout << "\n\n";
	return probabilityMatrix;
}


//prints whether an instance was correctly classified
//void feedForwardSingleInstance(/*Eigen::MatrixXf instance*/)
//{
/*multiply by the weights and biases
get classification by max Index.
say if it was right or wrong classified.*/
//}






