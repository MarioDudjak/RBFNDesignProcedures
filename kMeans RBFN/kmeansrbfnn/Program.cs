using System;
using System.Collections.Generic;
using System.Text;
using Accord.Math;

namespace kmeansrbfnn
{
	class Program
	{
		static void Main(string[] args)
		{
			Random rnd = new Random(1);
			DataSet Iris = new DataSet(3, new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" });
			Iris.ImportData("iris.txt");
			//DataSet Iris = new DataSet(3, new string[] { "1", "2", "3" });
			//Iris.ImportData("seeds.txt");
			//DataSet Iris = new DataSet(2, new string[] { "1", "2" });
			//Iris.ImportData("parkinsons.txt");
			//DataSet Iris = new DataSet(2, new string[] { "1", "2" });
			//Iris.ImportData("ionosphere.txt");
			//DataSet Iris = new DataSet(3, new string[] { "1", "2", "3" });
			//Iris.ImportData("thyroid.txt");
			//DataSet Iris = new DataSet(2, new string[] { "0", "1" });
			//Iris.ImportData("banknote.txt");
			//DataSet Iris = new DataSet(6, new string[] { "1", "2", "3", "4", "5", "6" });
			//Iris.ImportData("glass.txt"); //Nein

			DataSet[] tve = Iris.splitDataset(rnd, new double[] { 80, 20 });
			DataSet s = tve[0];

			int k = 5;
			for (int j = 5; j < 21; j += 5)
			{
				double mcrt = 0;
				double mset = 0;
				for (int i = 0; i < 30; i++)
				{
					kMeans km = new kMeans(s);
					km.Run(j, 500, 0.001);
					double[][] c = km.getCenters();
					double[] w = Utilities.getWidts(c);

					RBFNN net = new RBFNN(c, w, tve[0]);
                    //net.setInitialWeights();
                    //net.trainGD(500, 0.1, 0.7, s);
                    net.PInvWeights();
					double mcr = net.misclassificationRate(tve[1]);
					double tmp = net.sumSquaredError(s);
					double mse = tmp / s.Size / s.Dimensionality;

					if (double.IsNaN(mse))
						Console.WriteLine("Sranje: {0}, {1}, {2}", s.Size, s.Dimensionality, tmp);

					//Console.WriteLine("sse = {0}, mcr = {1}, mse = {2}, k = {3}", km.SSE, mcr, mse, i);

					mcrt += mcr;
					mset += mse;
				}

				Console.WriteLine("k = {0}\tavg.mcr = {1}\t\tavg.mse = {2}", j, mcrt / 30, mset / 30);
			}
		}
	}
}
