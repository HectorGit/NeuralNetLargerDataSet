#pragma once

/*
==============================================================================

NeuralNetwork.h
Created: 29 Sep 2017 2:54:03pm
Author:  Owner

==============================================================================
*/

#pragma once
#include "stdafx.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <iostream>


class NeuralNetwork {

public:
	NeuralNetwork(std::vector<int> &paramSizes);
	~NeuralNetwork();
	int mini_batch_size;

	void stochasticGradientDescent(/*Eigen::MatrixXf &trainingData, int epochs, int miniBatchSize, float eta, Eigen::MatrixXf &testData*/);
	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> all_Xs;
	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> all_Ys; // the assigned classification 
																					//is where there is a 1 (only one, in the vector).
	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> dummyTestDataSet_Xs;
	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> dummyTestDataSet_Ys;
	void classify(int numberOfAttributes, std::vector<int> &paramSizes/*technically should receive a std::vector of Matrix, but it's hard so avoided passing it*/); //prints statistic info
																												//void feedForwardSingleInstance(/*Eigen::MatrixXf instance*/); //prints whether an instance was correctly classified
																												//and what classification it was assigned
private:
	int epochs;
	int numLayers;
	std::vector<int> sizes;
	float eta;

	//GLOBAL - check which ones we need...
	Eigen::MatrixXf testData;
	Eigen::MatrixXf validationData;
	Eigen::MatrixXf trainingData;

	Eigen::MatrixXf activationL1;
	Eigen::MatrixXf activationL2;

	Eigen::MatrixXf x; //includes all the input vectors

	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> mini_Batch_Xs;
	std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> mini_Batch_Ys;
	Eigen::MatrixXf allGradientsBs[10][2]; //nabla_b
	Eigen::MatrixXf allGradientsWs[10][2]; //nabla_w
	Eigen::MatrixXf allActivations[10][3];
	Eigen::MatrixXf allZs[10][2];

	//LEVEL 1
	Eigen::MatrixXf weightsMatrixL1;
	Eigen::MatrixXf gradientWsL1;
	Eigen::MatrixXf biasesMatrixL1;
	Eigen::MatrixXf gradientBsL1;
	Eigen::MatrixXf zL1;

	//LEVEL 2
	Eigen::MatrixXf weightsMatrixL2;
	Eigen::MatrixXf gradientWsL2;
	Eigen::MatrixXf biasesMatrixL2;
	Eigen::MatrixXf gradientBsL2;
	Eigen::MatrixXf zL2;

	//=================================================================================

	void backPropagation(int mini_batch_size);
	void updateMiniBatch(/*Eigen::MatrixXf &mini_batch, float eta, Eigen::MatrixXf &biasesMatrixL1, Eigen::MatrixXf &weightsMatrixL1, Eigen::MatrixXf &biasesMatrixL2, Eigen::MatrixXf &weightsMatrixL2*/);
	void feedForward(Eigen::MatrixXf &x, Eigen::MatrixXf &y);
	int evaluate(Eigen::MatrixXf &activationL2, Eigen::MatrixXi &y);
	Eigen::MatrixXf costDerivative(Eigen::MatrixXf &outputActivations, Eigen::MatrixXf &y);
	Eigen::MatrixXf sigmoid_Vectorial(Eigen::MatrixXf &z);
	Eigen::MatrixXf sigmoid_Prime_Vectorial(Eigen::MatrixXf &z);
	void countErrors(int index, Eigen::MatrixXf &counters, Eigen::MatrixXf &classifications, Eigen::MatrixXf &yS);

	float calculateAccuracy(int total, Eigen::MatrixXf &counters);
	Eigen::MatrixXf obtainConfusionMatrix(int total, Eigen::MatrixXf &counters);

};