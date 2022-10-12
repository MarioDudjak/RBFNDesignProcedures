using System;
using System.Collections.Generic;
using System.Text;
using System.IO;

namespace ParticleSwarmOptimization
{
    class Program
    {
        static void Main(string[] args)
        {

			Console.ForegroundColor = ConsoleColor.White;
			
			/* Deklaracije: */
            Random rnd		= new Random(1);
			int runsNumber	= 30;
			String path		= "Datasets\\";
			StreamWriter wr = new StreamWriter(path + "rez.csv");
			wr.AutoFlush	= true;
			String dFormat	= "{0:0.0000}";

			/*PODACI: */
			#region data



			#region Datasets definition
			DataSet Banknote = new DataSet(2, new string[] { "0", "1" });
			Banknote.ImportData(path + "banknote.txt");
			DataSet Cardiotochography = new DataSet(10, new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" });
			Cardiotochography.ImportData(path + "cardiotochography.txt");
			DataSet CMC = new DataSet(3, new string[] { "1", "2", "3" });
			CMC.ImportData(path + "cmc.txt");
			DataSet Glass = new DataSet(6, new string[] { "1", "2", "3", "4", "5", "6" });
			Glass.ImportData(path + "glass.txt");
			DataSet HeartDisease = new DataSet(5, new string[] { "1", "2", "3", "4", "5" });
			HeartDisease.ImportData(path + "heart_disease.txt");
			DataSet ILPD = new DataSet(2, new string[] { "1", "2" });
			ILPD.ImportData(path + "ILPD.txt");
			DataSet Imageseg = new DataSet(7, new string[] { "1", "2", "3", "4", "5", "6", "7" });
			Imageseg.ImportData(path + "imageseg.txt");
			DataSet Ionosphere = new DataSet(2, new string[] { "1", "2" });
			Ionosphere.ImportData(path + "ionosphere.txt");
			DataSet Iris = new DataSet(3, new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" });
			Iris.ImportData(path + "iris.txt");
			DataSet Liver = new DataSet(2, new string[] { "1", "2" });
			Liver.ImportData(path + "liver.txt");
			DataSet Parkinsons = new DataSet(2, new string[] { "1", "2" });
			Parkinsons.ImportData(path + "parkinsons.txt");
			DataSet Pima = new DataSet(2, new string[] { "1", "2" });
			Pima.ImportData(path + "pima.txt");
			DataSet Seeds = new DataSet(3, new string[] { "1", "2", "3" });
			Seeds.ImportData(path + "seeds.txt");
			DataSet Statlog = new DataSet(4, new string[] { "1", "2", "3", "4" });
			Statlog.ImportData(path + "statlog.txt");
			DataSet Thoracic = new DataSet(2, new string[] { "0", "1" });
			Thoracic.ImportData(path + "thoracic.txt");
			DataSet Thyroid = new DataSet(3, new string[] { "1", "2", "3" });
			Thyroid.ImportData(path + "thyroid.txt");
			DataSet Vowel = new DataSet(6, new string[] { "1", "2", "3", "4", "5", "6" });
			Vowel.ImportData(path + "vowel.txt");
			DataSet Wbc = new DataSet(2, new string[] { "2", "4" });
			Wbc.ImportData(path + "wbc.txt");
			DataSet Wine = new DataSet(3, new string[] { "1", "2", "3" });
			Wine.ImportData(path + "wine.txt");
			DataSet Yeast = new DataSet(10, new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" });
			Yeast.ImportData(path + "yeast.txt");
			#endregion

			
			DataSet[] datasets = new DataSet[]
			{
				Iris, Cardiotochography, CMC, Glass, HeartDisease, ILPD, Imageseg, Ionosphere, Banknote, Liver,
				Parkinsons, Pima, Seeds, Statlog, Thoracic, Thyroid, Vowel, Wbc, Wine, Yeast
			};

			#endregion

			/* Vrtnja algoritma nad svim skupovima
			 * Ponavlja se runsNumber puta, u svakom runu se skup dijeli iznova
			 * Zapisuje u datoteku: mcr, mse, k, 
			*/
			for (int set = 0; set < datasets.Length; set++)
			{
				double avgMCR = 0, avgMSE = 0, avgkSize = 0;

				int classes = datasets[set].Classes;
				double[][] ConfusionMatrix = new double[classes][];
				for (int i = 0; i < classes; i++)
					ConfusionMatrix[i] = new double[classes];
			
				for (int run = 0; run < runsNumber; run++)
				{
					DataSet[] current = datasets[set].splitDataset(rnd, new double[] { 80, 20 });
					
					PSO pso = new PSO(current[0]);
					pso.Run(30, 0.9, 2, 2, 1000, 2, 20);

					double[][] c	= pso.getCenters();
					double[] w		= pso.getWidths();

					double MCR		= 0, MSE = 0, kSize = 0;

					RBFNN net		= new RBFNN(c, w, current[0]);
					net.PInvWeights();

					MCR		= net.misclassificationRate(current[1]);
					MSE		= net.sumSquaredError(current[1]) / current[1].Size;
					kSize	= c.Length;

					avgMCR		+= MCR;
					avgMSE		+= MSE;
					avgkSize	+= kSize;

					/* Zapis trenutnih rezultata u csv i na konzolu */
					wr.WriteLine(datasets[set].Name + "," + MCR + "," + MSE + "," + kSize + ",");
					Console.WriteLine(datasets[set].Name + "\t" + String.Format(dFormat, MCR) + "\t"
						+ String.Format(dFormat, MSE) + "\t" + kSize + "\t");

					/* TESTIRANJE CONFUSION MATRIXA */
					for (int pattern = 0; pattern < current[1].Size; pattern++)
					{
						double[] output = net.calcOutput(current[1][pattern]);
						int reallabel = current[1].DataLables[pattern];
						int label = determineLabel(output);
						ConfusionMatrix[reallabel][label]++;
					}
				}

				#region operacijeNadMatricom
				double total = 0;
				double[] sumRows = new double[classes];
				double[] sumCols = new double[classes];

				for (int i = 0; i < classes; i++)
				{
					for (int j = 0; j < classes; j++)
					{
						total += ConfusionMatrix[i][j];
						sumRows[i] += ConfusionMatrix[i][j];
						sumCols[i] += ConfusionMatrix[j][i];
					}
				}
				//printArray(sumRows, classes);
				//printArray(sumCols, classes);				
				double oACC = 0;
				double[] predictedAcc = new double[classes];

				// Precision & Recall:
				double precision = 0;
				double recall = 0;
				for (int i = 0; i < classes; i++)
				{
					oACC += ConfusionMatrix[i][i];
					precision += ConfusionMatrix[i][i] / sumRows[i];
					recall += ConfusionMatrix[i][i] / sumCols[i];
				}

				// Kappa:
				double pACC = 0;
				for (int i = 0; i < classes; i++)
				{
					predictedAcc[i] = sumRows[i] * sumCols[i] / total;
					pACC += predictedAcc[i];
				}

				pACC /= total;
				oACC /= total;
				double kappa = (oACC - pACC) / (1 - pACC);

				Console.WriteLine("__________________________________");
				Console.WriteLine("Precision:\t" + String.Format(dFormat, precision / 3));
				Console.WriteLine("Recall:\t\t" + String.Format(dFormat, recall / 3));
				Console.WriteLine("Pre. Acc.:\t" + String.Format(dFormat, pACC));
				Console.WriteLine("Obs. Acc.:\t" + String.Format(dFormat, oACC));
				Console.WriteLine("Kappa:\t\t" + String.Format(dFormat, kappa));
				#endregion

				wr.WriteLine(",,,," + kappa);

				Console.WriteLine("Confusion Matrix:");
				printMatrix(ConfusionMatrix, classes, classes);
				/* Ispis prosjeka na konzolu */
				Console.ForegroundColor = ConsoleColor.Cyan;
				Console.WriteLine(datasets[set].Name + "\t" + String.Format(dFormat, avgMCR / runsNumber) + "\t"
						+ String.Format(dFormat, avgMSE / runsNumber) + "\t" + avgkSize / runsNumber + "\t");
				Console.ForegroundColor = ConsoleColor.White;
				Console.WriteLine("__________________________________");



			}


			/*
			Random rnd = new Random();

			#region data

			String path = "Datasets\\";

			#region Datasets definition
				DataSet Banknote = new DataSet(2, new string[] { "0", "1" });
				Banknote.ImportData(path + "banknote.txt");
				DataSet Cardiotochography = new DataSet(10, new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" });
				Cardiotochography.ImportData(path + "cardiotochography.txt");
				DataSet CMC = new DataSet(3, new string[] { "1", "2", "3" });
				CMC.ImportData(path + "cmc.txt");
				DataSet Glass = new DataSet(6, new string[] { "1", "2", "3", "4", "5", "6" });
				Glass.ImportData(path + "glass.txt");
				DataSet HeartDisease = new DataSet(5, new string[] { "1", "2", "3", "4", "5" });
				HeartDisease.ImportData(path + "heart_disease.txt");
				DataSet ILPD = new DataSet(2, new string[] { "1", "2" });
				ILPD.ImportData(path + "ILPD.txt");
				DataSet Imageseg = new DataSet(7, new string[] { "1", "2", "3", "4", "5", "6", "7" });
				Imageseg.ImportData(path + "imageseg.txt");
				DataSet Ionosphere = new DataSet(2, new string[] { "1", "2" });
				Ionosphere.ImportData(path + "ionosphere.txt");
				DataSet Iris = new DataSet(3, new string[] { "Iris-setosa", "Iris-versicolor", "Iris-virginica" });
				Iris.ImportData(path + "iris.txt");
				DataSet Liver = new DataSet(2, new string[] { "1", "2" });
				Liver.ImportData(path + "liver.txt");
				DataSet Parkinsons = new DataSet(2, new string[] { "1", "2" });
				Parkinsons.ImportData(path + "parkinsons.txt");
				DataSet Pima = new DataSet(2, new string[] { "1", "2" });
				Pima.ImportData(path + "pima.txt");
				DataSet Seeds = new DataSet(3, new string[] { "1", "2", "3" });
				Seeds.ImportData(path + "seeds.txt");
				DataSet Statlog = new DataSet(4, new string[] { "1", "2", "3", "4" });
				Statlog.ImportData(path + "statlog.txt");
				DataSet Thoracic = new DataSet(2, new string[] { "0", "1" });
				Thoracic.ImportData(path + "thoracic.txt");
				DataSet Thyroid = new DataSet(3, new string[] { "1", "2", "3" });
				Thyroid.ImportData(path + "thyroid.txt");
				DataSet Vowel = new DataSet(6, new string[] { "1", "2", "3", "4", "5", "6" });
				Vowel.ImportData(path + "vowel.txt");
				DataSet Wbc = new DataSet(2, new string[] { "2", "4" });
				Wbc.ImportData(path + "wbc.txt");
				DataSet Wine = new DataSet(3, new string[] { "1", "2", "3" });
				Wine.ImportData(path + "wine.txt");
				DataSet Yeast = new DataSet(10, new string[] { "1", "2", "3", "4", "5", "6", "7", "8", "9", "10" });
				Yeast.ImportData(path + "yeast.txt");
			#endregion

			DataSet[] datasets = new DataSet[20]
			{
				Iris, Cardiotochography, CMC, Glass, HeartDisease, ILPD, Imageseg, Ionosphere, Banknote, Liver,
				Parkinsons, Pima, Seeds, Statlog, Thoracic, Thyroid, Vowel, Wbc, Wine, Yeast
			};
			#endregion

			StreamWriter wr = new StreamWriter(path+"rez.csv");

			for (int i = 0; i < datasets.Length; i++) 
			{
				Console.Write(datasets[i].Name + "\t");
				wr.Write(datasets[i].Name + "\t");

				DataSet[] tve = datasets[i].splitDataset(rnd, new double[] { 80, 20 });
				PSO pso			= new PSO(tve[0]);

				pso.Run(20, 0.9, 2, 2, 1000, 2, 20);

				double[][] c	= pso.getCenters();
				double[] w		= pso.getWidths();

				RBFNN net		= new RBFNN(c, w, tve[0]);
				net.PInvWeights();

				double mcr = net.misclassificationRate(tve[1]);
				double mse = net.sumSquaredError(tve[1]) / tve[1].Size;
				Console.WriteLine(mcr + "\t" + mse + "\t" + w.Length);
				wr.WriteLine(mcr + "\t" + w.Length);
			}

			wr.Close();
			 * */
        }

		public static void printMatrix(double [][] matrix, int r, int c)
		{
			for (int i = 0; i < r; i++)
			{
				for (int j = 0; j < c; j++)
				{
					Console.Write(matrix[i][j] + "\t");
				}
				Console.WriteLine();
			}
			Console.WriteLine("__________________________________");
		}
		public static void printArray(double[] array, int elements)
		{
			for (int i = 0; i < elements; i++)
			{
				Console.Write(array[i] + "\t");
			}
			Console.WriteLine();
		}
		public static int determineLabel(double[] output)
		{
			int label = 0;
			for (int i = 1; i < output.Length; i++)
			{
				if (output[i] > output[label])
					label = i;
			}
			return label;
		}
    }
}
