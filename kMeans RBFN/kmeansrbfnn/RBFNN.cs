using System;
using System.Collections.Generic;
using System.Text;
using Accord.Math;

namespace kmeansrbfnn
{
	class RBFNN
	{
		int kappa;
		int dim;
		int nClasses;
		double bias;

		double[][] weights;
		double[][] weightchanges;
		double[][] centers;
		double[] widths;
		double[] hOutputs;
		DataSet data;

		//CTOR
		public RBFNN(double[][] centers, double[] widths, DataSet data)
		{
			this.data = data;
			this.centers = centers;
			this.widths = widths;
			bias = 0.0;
			nClasses = data.Classes;
			dim = data.Dimensionality;
			kappa = centers.Length;
			weights = createMatrix(kappa, nClasses);
			weightchanges = createMatrix(kappa, nClasses);
			hOutputs = new double[kappa];
		}

		//CLASSIFICATION INTERFEJS
		double[] calcOutput(double[] inputVector)
		{
			double[] outputVector = new double[nClasses];
			double hSum = 0.0;
			for (int i = 0; i < kappa; i++)
			{
				hOutputs[i] = gaussianFunction(inputVector, centers[i], widths[i]);
				hSum += hOutputs[i];
			}

			for (int i = 0; i < kappa; i++)
			{
				hOutputs[i] /= hSum;
			}
			for (int i = 0; i < nClasses; i++)
			{
				for (int j = 0; j < kappa; j++)
				{
					outputVector[i] += hOutputs[j] * weights[j][i];
				}
				outputVector[i] += bias;
                //outputVector[i] = sigmoidFunction(outputVector[i]);
			}

			return outputVector;
		}

		public double[,] calcHoutputs()
		{
			int m = data.Size;
			double[,] hOut = new double[m, kappa];
			for (int i = 0; i < m; i++)
			{
				double hSum = 0;
				for (int j = 0; j < kappa; j++)
				{
					hOut[i, j] = gaussianFunction(data[i], centers[j], widths[j]);
					hSum += hOut[i, j];
				}

                for (int j = 0; j < kappa; j++)
                {
                    hOut[i, j] /= hSum;
                }
			}

			return hOut;
		}

		public double[,] calcDesired()
		{
			int m = data.Size;
			double[,] desired = new double[m, nClasses];
            //for (int i = 0; i < m; i++)
            //    for (int j = 0; j < nClasses; j++)
            //        desired[i, j] = 0.1;

			for (int i = 0; i < m; i++)
				desired[i, data.DataLables[i]] = 1;

			return desired;
		}

		public void PInvWeights()
		{
			double[,] w;
			double[,] h = calcHoutputs();
			double[,] d = calcDesired();

			double[,] t = Matrix.PseudoInverse(h);
			w = Matrix.Multiply(t, d);

			for (int i = 0; i < kappa; i++)
				for (int j = 0; j < nClasses; j++)
					weights[i][j] = w[i, j];
		}

		//TRENING INTERFEJS
		public void trainGD(int epochs, double lrn, double mom, DataSet d)
		{
			data = d;
			for (int i = 0; i < epochs; i++)
			{
				for (int j = 0; j < data.Size; j++)
				{
					double[] outputVector = this.calcOutput(data[j]);
					double[] desiredOutput = new double[nClasses];
					for (int l = 0; l < desiredOutput.Length; l++)
						desiredOutput[l] = 0.1;
					desiredOutput[data.DataLables[j]] = 0.9;
					for (int k = 0; k < kappa; k++)
					{
						for (int l = 0; l < nClasses; l++)
						{
							double temp = weights[k][l];
							weights[k][l] += -lrn * (desiredOutput[l] - outputVector[l]) * sigmoidDerivativeFunction(outputVector[l])
								* hOutputs[k] + mom * weightchanges[k][l];
							weightchanges[k][l] = weights[k][l] - temp;
						}
					}
				}
			}

		}


		//HELPER INTERFEJS
		private double gaussianFunction(double[] v1, double[] v2, double width)
		{
			double y = 0;
			y = euclideanDistance(v1, v2);
			y *= y;
			y /= -2 * width * width;
			y = Math.Exp(y);

			return y;
		}
		private double euclideanDistance(double[] v1, double[] v2)
		{
			double result = 0;
			for (int i = 0; i < v1.Length; i++)
			{
				result += (v1[i] - v2[i]) * (v1[i] - v2[i]);
			}
			return Math.Sqrt(result);
		}
		private double[][] createMatrix(int r, int c)
		{
			double[][] result = new double[r][];
			for (int i = 0; i < r; i++)
			{
				result[i] = new double[c];
			}
			return result;
		}
		private double sigmoidFunction(double value)
		{
			double sig = 0.0;
			sig = 1 / (1 + Math.Exp(value));
			return sig;
		}
		public double sigmoidDerivativeFunction(double value)
		{
			double dsig = 0.0;
			double sig = sigmoidFunction(value);
			dsig = sig * (1 - sig);
			return dsig;
		}
		public void setInitialWeights(Random rnd)
		{
			for (int i = 0; i < kappa; i++)
			{
				for (int j = 0; j < nClasses; j++)
				{
					weights[i][j] = -1 + rnd.NextDouble() * 2;
				}
			}
		}
		public void setInitialWeights(int seed)
		{
			Random rnd = new Random(seed);
			for (int i = 0; i < kappa; i++)
			{
				for (int j = 0; j < nClasses; j++)
				{
					weights[i][j] = -1 + rnd.NextDouble() * 2;
				}
			}
		}
		public void setInitialWeights()
		{
			Random rnd = new Random();
			for (int i = 0; i < kappa; i++)
			{
				for (int j = 0; j < nClasses; j++)
				{
					weights[i][j] = -1 + rnd.NextDouble() * 2;
				}
			}
		}
		public void setInitialWeights(double x)
		{
			for (int i = 0; i < kappa; i++)
			{
				for (int j = 0; j < nClasses; j++)
				{
					weights[i][j] = x;
				}
			}
		}
		public double sumSquaredError(DataSet d)
		{
			double sse = 0.0;
			double[] desiredOutput = new double[nClasses];
			double[] outputVector = null;
			for (int i = 0; i < d.Size; i++)
			{
				outputVector = this.calcOutput(d[i]);
                for (int j = 0; j < nClasses; j++)
                    desiredOutput[j] = 0.0;
				desiredOutput[d.DataLables[i]] = 1;

				for (int j = 0; j < nClasses; j++)
				{
					sse += (desiredOutput[j] - outputVector[j]) * (desiredOutput[j] - outputVector[j]);
				}
			}

			return sse;
		}
		public double misclassificationRate(DataSet d)
		{
			double mcr = 0.0;
			for (int i = 0; i < d.Size; i++)
			{
				double[] outputVector = calcOutput(d[i]);
				int label = 0;
				double max = outputVector[0];
				for (int j = 1; j < nClasses; j++)
				{
					if (outputVector[j] > max)
					{
						max = outputVector[j];
						label = j;
					}
				}
				if (label != d.DataLables[i])
				{
					mcr++;
				}
			}
			return mcr / d.Size;
		}

		public void setWeights(double[][] w)
		{
			weights = w;
		}

		public void setWeights(double[,] w)
		{
			for (int i = 0; i < kappa; i++)
				for (int j = 0; j < nClasses; j++)
					weights[i][j] = w[i, j];
		}

		public double[][] Weights { get { return weights; } }
	}
}
