/*
==============================================================================

Main.cpp
Created: 29 Sep 2017 3:10:51pm
Author:  Owner

==============================================================================
*/
#include "stdafx.h"
#include <Eigen/Core>
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <iostream>
#include "NeuralNetwork.h"

#include <iostream>
//#include <istringstream>
#include <sstream>
#include <fstream>
#include <string>

int main() {
	bool debug = false;
	Eigen::IOFormat CleanFmt(4, 0, ", ", "\n", "[", "]");
	std::vector<int> paramSizes = {9,12,6};
	NeuralNetwork* aNet = new NeuralNetwork(paramSizes);
	int totalNumberOfInstances = 200;
	int numberOfTrainingInstances = 180;       //change this to 100 later.
	int numberOfTestingInstances = 20; //haven't picked them out yet. 
	int numberOfAttributes = 10;
	Eigen::MatrixXf dummyMatrix = Eigen::MatrixXf::Zero(totalNumberOfInstances, numberOfAttributes);//does this mean we have 5 columns?

	int rowCounter = 0;
	int colCounter = 0;

	std::ifstream  data("glass.csv");

	std::string line;
	while (std::getline(data, line))
	{
		std::stringstream  lineStream(line);
		std::string        cell;
		while (std::getline(lineStream, cell, ','))
		{
			//std::cout << colCounter;

			std::istringstream iss(cell);
			float myVal;
			iss >> myVal;

			if (colCounter >= numberOfAttributes-1) {
				dummyMatrix(rowCounter, colCounter) = myVal;
				colCounter = 0;
				rowCounter++;
				//std::cout << "\n";
			}
			else {
				dummyMatrix(rowCounter, colCounter) = myVal;
				colCounter++;
			}

		}
	}

	if (debug) { std::cout << "\n this Matrix: \n" << dummyMatrix.format(CleanFmt); }

	
	for (int i = 0; i < numberOfTrainingInstances; i++) {

		Eigen::MatrixXf tempX = Eigen::MatrixXf::Zero(1, numberOfAttributes-1); //zeros matrix 1 by 4. or 2 by 4??? hmm.
		for (int j = 0; j < numberOfAttributes - 1; j++) {
			tempX(0, j) = dummyMatrix(i, j);
		}
		aNet->all_Xs.emplace_back(tempX.transpose());//because we want 4 by 1 !

		Eigen::MatrixXf tempY = Eigen::MatrixXf::Zero(paramSizes[2], 1);
		tempY(dummyMatrix(i, numberOfAttributes-1) - 1, 0) = 1; // holy f -> dummyMatrix at col numberOfAttributes (counting from 0) has values ... up to numberOfAttributes -1 
											 //need to -1. because the values read from the file are counting from 1.
											//classes 1 to 6.
		aNet->all_Ys.emplace_back(tempY);// no need to transpose
	}

	if (debug) { std::cout << "\n sample X Matrix: \n" << aNet->all_Xs[0].format(CleanFmt); }

	if (debug) { std::cout << "\n sample Y Matrix: \n" << aNet->all_Ys[0].format(CleanFmt); }

	printf(" \n--------------TRAINING--------------\n");
	printf(" calling stochastic gradient descent\n");

	aNet->stochasticGradientDescent();
	
	
	printf(" \n--------------TESTING--------------\n");
	printf("          calling classify          \n");

	for (int i = numberOfTrainingInstances; i < numberOfTrainingInstances + numberOfTestingInstances; i++) {

		//printf("\ndummy testing data %d : \n", i);

		Eigen::MatrixXf tempX = Eigen::MatrixXf::Zero(1, numberOfAttributes-1); //zeros matrix 1 by 4. or 2 by 4??? hmm.
		for (int j = 0; j < numberOfAttributes - 1; j++) {
			tempX(0, j) = dummyMatrix(i, j);
		}
		aNet->dummyTestDataSet_Xs.emplace_back(tempX.transpose());//because we want 4 by 1 !

		Eigen::MatrixXf tempY = Eigen::MatrixXf::Zero(paramSizes[2], 1);
		tempY(dummyMatrix(i, numberOfAttributes-1) - 1, 0) = 1; // holy f -> dummyMatrix at col 4 (counting from 0) has values 1, 2 ,3 
											 //need to -1.
		aNet->dummyTestDataSet_Ys.emplace_back(tempY);// no need to transpose
	}

	if (debug) { std::cout << "\n sample X Matrix: \n" << aNet->dummyTestDataSet_Xs[0].format(CleanFmt); }

	if (debug) { std::cout << "\n sample Y Matrix: \n" << aNet->dummyTestDataSet_Ys[0].format(CleanFmt); }

	aNet->classify(numberOfAttributes-1, paramSizes);
	
	std::cin.get();

	return 0;
}